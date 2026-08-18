# ==========================================
# gen_data_30m.jl
# ==========================================
using HTTP
using JSON
using DataFrames
using Dates
using TimeZones
using CSV
using Statistics

# ==========================================
# --- MAIN CONFIGURATION ---
# ==========================================
symbol = "SPY"
start_d = Date("2016-01-01")
end_d = Date("2026-03-31")

# --- ALPACA CREDENTIALS ---
if !haskey(ENV, "ALPACA_KEY") || !haskey(ENV, "ALPACA_SECRET")
  error("CRITICAL: Alpaca credentials not found. Ensure 'ALPACA_KEY' and 'ALPACA_SECRET' are set in your environment.")
end

const ALPACA_KEY = ENV["ALPACA_KEY"]
const ALPACA_SECRET = ENV["ALPACA_SECRET"]
const BASE_URL = "https://data.alpaca.markets/v2/stocks/bars"

# ==========================================
# 1. FETCH 10-MINUTE DATA
# ==========================================
function fetch_alpaca_data_10m(symbol::String, start_date::Date, end_date::Date)
  println("Fetching 10-minute data for $symbol from Alpaca ($start_date to $end_date)...")

  headers = [
    "APCA-API-KEY-ID" => ALPACA_KEY,
    "APCA-API-SECRET-KEY" => ALPACA_SECRET,
    "accept" => "application/json"
  ]

  all_bars = DataFrame()
  current_start = start_date
  page_token = nothing

  while current_start < end_date
    url = "$BASE_URL?symbols=$symbol&timeframe=10Min&adjustment=all&start=$(current_start)T00:00:00Z&end=$(end_date)T23:59:59Z&limit=10000"
    if !isnothing(page_token)
      url *= "&page_token=$page_token"
    end

    response = HTTP.get(url, headers)
    parsed = JSON.parse(String(response.body))

    if !haskey(parsed["bars"], symbol)
      break
    end

    bars = parsed["bars"][symbol]

    timestamps = [DateTime(replace(b["t"], "Z" => "")) for b in bars]
    closes = [Float64(b["c"]) for b in bars]
    volumes = [Float64(b["v"]) for b in bars]

    df_chunk = DataFrame(utc_time=timestamps, close=closes, volume=volumes)
    append!(all_bars, df_chunk)

    page_token = get(parsed, "next_page_token", nothing)
    if isnothing(page_token)
      break
    end

    println("  Fetched up to $(last(timestamps))")
    sleep(0.3)
  end

  ny_tz = tz"America/New_York"
  all_bars.ny_time = [astimezone(ZonedDateTime(dt, tz"UTC"), ny_tz) |> DateTime for dt in all_bars.utc_time]

  # Label each bar by the end of its interval (+ 10 minutes)
  all_bars.ny_time = all_bars.ny_time .+ Minute(10)

  sort!(all_bars, :ny_time)
  unique!(all_bars, :ny_time)

  return all_bars
end

# ==========================================
# 2. AGGREGATE TO CUSTOM 30-MINUTE WINDOWS
# ==========================================
function aggregate_to_30m(df_10m::DataFrame)
  println("Aggregating 10m bars into custom 30m windows (anchored at 09:50)...")

  # Map each 10m bar to its target 30m window
  function get_window_end(dt::DateTime)
    d = Date(dt)
    t = Time(dt)

    # The 09:50 anchor is kept in a window of its own
    if t <= Time(9, 50)
      return DateTime(d, Time(9, 50))
    end

    m = hour(t) * 60 + minute(t)
    target_m = 590 + ceil(Int, (m - 590) / 30) * 30
    return DateTime(d, Time(target_m ÷ 60, target_m % 60))
  end

  df = copy(df_10m)
  df.window_end = get_window_end.(df.ny_time)

  # Group and summarize the data
  df_30m = combine(groupby(df, :window_end)) do group
    DataFrame(
      utc_time=last(group.utc_time),
      close=last(group.close),
      volume=sum(group.volume)
    )
  end

  # Rename the grouping column back to the original name
  rename!(df_30m, :window_end => :ny_time)

  sort!(df_30m, :ny_time)
  return df_30m
end


# ==========================================
# 3. COMPUTE RETURNS AND MATCH FOMC
# ==========================================
function process_returns_and_fomc(price_data::DataFrame, fomc_data::DataFrame)
  println("Computing clean returns and tagging FOMC events...")

  # Compute returns
  price_data.log_return = [0.0; diff(log.(price_data.close))] * 100
  time_diffs = [Minute(30); Minute.(diff(price_data.ny_time))]
  price_data.real_gap_minutes = Dates.value.(time_diffs)

  # Drop rows with gaps > 30 min.
  # This automatically removes overnight returns and data holes.
  filter!(row -> row.real_gap_minutes <= 30, price_data)

  # Columns required by the model
  price_data.dt_minutes = fill(30.0, nrow(price_data))
  price_data.target_shock = zeros(nrow(price_data))
  price_data.is_fomc = falses(nrow(price_data))

  # Inject the FOMC shock
  for row in eachrow(fomc_data)
    f_time = row.start
    # Locate the 30m bar containing the announcement time
    idx = findfirst(t -> (t - Minute(30)) < f_time <= t, price_data.ny_time)
    if !isnothing(idx)
      price_data.is_fomc[idx] = true
      price_data.target_shock[idx] = row.target_shock
    end
  end

  # Standardize dt: 1 year = 252 days * 390 market minutes
  price_data.dt = price_data.dt_minutes ./ (252.0 * 390.0)

  # Drop the internal helper column
  select!(price_data, Not(:real_gap_minutes))

  return price_data
end

# ==========================================
# 4. TIME-OF-DAY (ToD) NORMALIZATION
# ==========================================
function apply_tod_normalization(data::DataFrame)
  println("Applying Time-of-Day (ToD) volatility normalization...")

  data.time_of_day = Time.(data.ny_time)
  clean_bars = filter(row -> row.dt_minutes == 30.0 && !row.is_fomc, data)

  tod_profile = combine(groupby(clean_bars, :time_of_day), :log_return => var => :raw_tod_var)
  mean_tod_var = mean(tod_profile.raw_tod_var)
  tod_profile.norm_tod_factor = sqrt.(tod_profile.raw_tod_var ./ mean_tod_var)

  data = leftjoin(data, tod_profile, on=:time_of_day)
  data.norm_tod_factor = coalesce.(data.norm_tod_factor, 1.0)
  data.adj_log_return = data.log_return ./ data.norm_tod_factor

  select!(data, Not(:time_of_day))
  return data
end

# ==========================================
# 5. EXECUTION
# ==========================================
function is_early_close(d::Date)
  if month(d) == 7 && day(d) == 3
    return true
  elseif month(d) == 12 && day(d) == 24
    return true
  elseif month(d) == 11 && dayofweek(d) == Friday && 23 <= day(d) <= 29
    return true
  end
  return false
end

# Load monetary policy surprises
fomc_data = DataFrame(CSV.File("../data/fomc_surprises_jk.csv"))
cleaned_dates = [first(replace(string(d), "T" => " "), 16) for d in fomc_data.start]
fomc_data.start = DateTime.(cleaned_dates, dateformat"yyyy-mm-dd HH:MM")
fomc_data = fomc_data[occursin.("FOMC Rate Decision (Scheduled)", fomc_data.Event), :]
select!(fomc_data, :start, :MP1 => :target_shock)

# Run the pipeline
mkpath("../data/$symbol")
raw_file_path = "../data/$symbol/$(symbol)_10m_raw.csv"

if isfile(raw_file_path)
  println("✅ Raw file found. Loading data from $raw_file_path...")
  raw_10m_data = CSV.read(raw_file_path, DataFrame)
else
  raw_10m_data = fetch_alpaca_data_10m(symbol, start_d, end_d)
  println("Saving downloaded raw data to $raw_file_path...")
  CSV.write(raw_file_path, raw_10m_data)
end

println("Filtering for Adjusted Regular Trading Hours...")
# Keep bars from 09:50 (anchor) through 15:50 (last usable bar)
first_valid_bar = Time(9, 50)
market_close = Time(15, 50)
early_close = Time(12, 50)

filter!(row -> begin
    t = Time(row.ny_time)
    d = Date(row.ny_time)
    if is_early_close(d)
      return first_valid_bar <= t <= early_close
    else
      return first_valid_bar <= t <= market_close
    end
  end, raw_10m_data)

println("Rows remaining after filter: ", nrow(raw_10m_data))

# Transformations
df_30m = aggregate_to_30m(raw_10m_data)
df_30m = df_30m[2:end, :]
df_30m = process_returns_and_fomc(df_30m, fomc_data)
df_30m = apply_tod_normalization(df_30m)

display(size(df_30m))
display(last(df_30m, 20))

# Save
CSV.write("../data/$symbol/$(symbol)_30m.csv", df_30m)
println("\n✅ Intraday 30m dataset for $symbol complete. Ready for PMCMC estimation.")

