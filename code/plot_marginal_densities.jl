# ==========================================
# plot_posterior_densities.jl
# ==========================================
using CSV
using DataFrames
using Plots
using StatsPlots
using Printf
using Statistics
using LaTeXStrings
using Measures

# ==========================================
# --- CONFIGURACIÓN PRINCIPAL ---
# ==========================================
symbols = ["SPY", "IWM"]

# ==========================================
# 1. FIND AND LOAD ALL CHAIN FILES
# ==========================================
symbols_data = Dict{String,DataFrame}()

for sym in symbols
  println("Scanning directory for $sym chain files...")
  chains_dir = "../mcmc_chains/$sym/"

  if !isdir(chains_dir)
    println("WARNING: Directory $chains_dir does not exist. Skipping $sym.")
    continue
  end

  all_files = readdir(chains_dir)
  chain_files = filter(f -> startswith(f, "chain_") && endswith(f, ".csv"), all_files)

  if isempty(chain_files)
    println("WARNING: No chain files found for $sym. Skipping.")
    continue
  end

  println("Found $(length(chain_files)) chain files for $sym:")
  chains = Vector{DataFrame}()
  for file in chain_files
    println(" -> Loading: $file")
    df = CSV.read(joinpath(chains_dir, file), DataFrame)
    push!(chains, df)
  end

  # Pool all chains for this symbol into a single DataFrame
  pooled_df = vcat(chains...)

  # Compute the new derived quantities based on the analytical properties
  # Multiply by 252 to convert the half-life from years to trading days
  pooled_df.tau_half = (log(2.0) ./ pooled_df.kappa) .* 252.0

  pooled_df.E_sqrt_V = exp.((pooled_df.theta ./ 2.0) .+ (pooled_df.sigma_v .^ 2 ./ (16.0 .* pooled_df.kappa)))
  pooled_df.Std_sqrt_V = pooled_df.E_sqrt_V .* sqrt.(exp.(pooled_df.sigma_v .^ 2 ./ (8.0 .* pooled_df.kappa)) .- 1.0)

  # Compute percentage increments for volatility jumps
  pooled_df.inc_alpha_0 = 100.0 .* (exp.(pooled_df.alpha_0 ./ 2.0) .- 1.0)
  pooled_df.inc_alpha_1 = 100.0 .* (exp.(pooled_df.alpha_1 .* 0.25 ./ 2.0) .- 1.0)

  symbols_data[sym] = pooled_df
end

if isempty(symbols_data)
  println("ERROR: No valid data loaded for any symbol.")
  exit()
end

# ==========================================
# 2. DEFINE PARAMETER GROUPS & LABELS
# ==========================================
cont_params = ["mu", "tau_half", "E_sqrt_V", "Std_sqrt_V", "rho"]
jump_params = ["beta_0", "beta_1", "sigma_eps", "inc_alpha_0", "inc_alpha_1"]

latex_names = Dict(
  "mu" => L"$\mu \quad [\%/\mathrm{año}]$",
  "tau_half" => L"$\tau_{1/2} \quad [\mathrm{días}]$",
  "E_sqrt_V" => L"$\mathrm{E}\left[\sqrt{V_t}\right] \quad [\%/\mathrm{año}]$",
  "Std_sqrt_V" => L"$\mathrm{Std}\left[\sqrt{V_t}\right] \quad [\%/\mathrm{año}]$",
  "rho" => L"$\rho$",
  "beta_0" => L"$\beta_0 \quad [\mathrm{p.p.}]$",
  "beta_1" => L"$\beta_1 \quad [\mathrm{p.p.}/\mathrm{p.p.}]$",
  "sigma_eps" => L"$\sigma_\epsilon \quad [\mathrm{p.p.}]$",
  "inc_alpha_0" => L"$\mathrm{Incremento} \ \mathrm{base} \ \mathrm{en} \ \sqrt{V_t} \quad [\%]$",
  "inc_alpha_1" => L"$\mathrm{Incremento} \ \mathrm{marginal} \ \mathrm{en} \ \sqrt{V_t} \quad [\%]$"
)

# ==========================================
# 3. PLOTTING & SUMMARY FUNCTIONS
# ==========================================
default(
  grid=false,
  dpi=200
)
scalefontsizes(1.5)

fill_colors = [:cyan3, :orange2, :purple3, :green3]
line_colors = [:cyan4, :darkorange, :indigo, :darkgreen]

function generate_centered_figure(sym_data_dict, symbols_list, param_list, latex_map, output_name)
  println("\nGenerating plot for: $output_name")

  plots_array = []

  for (p_idx, param) in enumerate(param_list)
    show_legend = (p_idx == 1)
    display_name = get(latex_map, param, param)

    p_dens = plot(title="",
      xlabel=display_name,
      yticks=false,
      ylabel="",
      yaxis=false,
      legend=show_legend ? :topleft : false,
      bottom_margin=6Plots.mm)

    for (s_idx, sym) in enumerate(symbols_list)
      if haskey(sym_data_dict, sym) && param in names(sym_data_dict[sym])

        pooled_data = sym_data_dict[sym][!, param]

        c_fill = fill_colors[mod1(s_idx, length(fill_colors))]
        c_line = line_colors[mod1(s_idx, length(line_colors))]

        density!(p_dens, pooled_data,
          bandwidth=0.2 * std(pooled_data),
          fillcolor=c_fill, fill=true, fillalpha=0.25,
          linecolor=c_line, linewidth=2, label=sym)
      end
    end

    push!(plots_array, p_dens)
  end

  emp = plot(framestyle=:none, grid=false, showaxis=false, ticks=false, legend=false)

  custom_layout = @layout [
    a{0.333w} b{0.333w} c
    d{0.166w} e{0.333w} f{0.333w} g
  ]

  final_plots = [
    plots_array[1], plots_array[2], plots_array[3],
    emp, plots_array[4], plots_array[5], emp
  ]

  final_plot = plot(final_plots..., layout=custom_layout, size=(1200, 500))

  savefig(final_plot, output_name)
  println("Saved successfully: $output_name")
end

function print_summary_table(sym_data_dict, symbols_list, param_list, table_title)
  for sym in symbols_list
    if !haskey(sym_data_dict, sym)
      continue
    end

    pooled_data_df = sym_data_dict[sym]

    println("\n===================================================================================================")
    println(" POSTERIOR SUMMARY: $table_title ($sym)")
    println("===================================================================================================")
    @printf("%-15s | %-9s | %-9s | %-9s | %-9s | %-9s | %-9s | %-9s\n",
      "Parameter", "Mean", "Std Dev", "5%", "25%", "50%", "75%", "95%")
    println("-"^102)

    for param in param_list
      if !(param in names(pooled_data_df))
        continue
      end

      data_col = pooled_data_df[!, param]

      p_mean = mean(data_col)
      p_std = std(data_col)
      q05 = quantile(data_col, 0.05)
      q25 = quantile(data_col, 0.25)
      q50 = quantile(data_col, 0.50)
      q75 = quantile(data_col, 0.75)
      q95 = quantile(data_col, 0.95)

      @printf("%-15s | %9.4f | %9.4f | %9.4f | %9.4f | %9.4f | %9.4f | %9.4f\n",
        param, p_mean, p_std, q05, q25, q50, q75, q95)
    end
    println("===================================================================================================\n")
  end
end

# ==========================================
# 4. EXECUTE AND SAVE
# ==========================================
mkpath("../figs")

# Continuous Parameters
generate_centered_figure(symbols_data, symbols, cont_params, latex_names, "../figs/posterior_continuous_all.pdf")
print_summary_table(symbols_data, symbols, cont_params, "CONTINUOUS PARAMETERS")

# Jump Parameters
generate_centered_figure(symbols_data, symbols, jump_params, latex_names, "../figs/posterior_jumps_all.pdf")
print_summary_table(symbols_data, symbols, jump_params, "JUMP PARAMETERS")
