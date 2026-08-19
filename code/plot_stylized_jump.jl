# ==========================================
# plot_stylized_jump.jl
# ==========================================
using CSV
using DataFrames
using Statistics
using Distributions
using Random
using Plots
using StatsPlots
using LaTeXStrings
using Measures # Needed for the subplot margin adjustments

# ==========================================
# --- MAIN CONFIGURATION ---
# ==========================================
theme(:bright)
symbols = ["SPY", "IWM"]
subplots = []

# Fixed seed: the predictive band is simulated, so this keeps the figure
# reproducible across runs
Random.seed!(20240101)

println("Starting generation of the combined figure...")

# Shared y-axis range so the two panels are directly comparable
y_limits = (-2.1, 2.4)

for (idx, symbol) in enumerate(symbols)
  # ==========================================
  # 1. LOAD DATA & CHAINS
  # ==========================================
  println("Loading data for $symbol...")
  master_data = CSV.read("../data/$symbol/$(symbol)_30m.csv", DataFrame)

  # Keep only the FOMC event windows
  fomc_events = filter(row -> row.is_fomc == true, master_data)
  observed_shocks = fomc_events.target_shock
  actual_returns = fomc_events.adj_log_return

  println("Loading MCMC chains for $symbol...")
  chains_dir = "../mcmc_chains/$symbol/"

  if !isdir(chains_dir)
    error("Directory $chains_dir does not exist. Run run_pmcmc.jl first.")
  end

  all_files = readdir(chains_dir)
  chain_files = chains_dir .* filter(f -> startswith(f, "chain_") && endswith(f, ".csv"), all_files)

  if isempty(chain_files)
    error("No chain files found in $chains_dir. Run run_pmcmc.jl first.")
  end

  chains_array = [CSV.read(f, DataFrame) for f in chain_files]
  chain_df = vcat(chains_array...)

  # Extract the jump-equation parameters
  beta_0_samples = chain_df.beta_0
  beta_1_samples = chain_df.beta_1
  sigma_eps_samples = chain_df.sigma_eps
  N_samples = length(beta_0_samples)

  # ==========================================
  # 2. COMPUTE PREDICTIVE DISTRIBUTIONS
  # ==========================================
  println("Computing predictive intervals for $symbol...")

  min_shock = minimum(observed_shocks) - 0.02
  max_shock = maximum(observed_shocks) + 0.02
  shock_grid = range(min_shock, max_shock, length=100)

  median_expected_jump = zeros(length(shock_grid))
  pred_lower_95 = zeros(length(shock_grid))
  pred_upper_95 = zeros(length(shock_grid))

  for (i, x) in enumerate(shock_grid)
    # Conditional mean implied by each posterior draw
    expected_means = beta_0_samples .+ beta_1_samples .* x

    # Posterior predictive: add the idiosyncratic announcement noise, so the
    # band covers realized returns rather than just the regression line
    noise = randn(N_samples) .* sigma_eps_samples
    simulated_realizations = expected_means .+ noise

    median_expected_jump[i] = median(expected_means)
    pred_lower_95[i] = quantile(simulated_realizations, 0.025)
    pred_upper_95[i] = quantile(simulated_realizations, 0.975)
  end

  # ==========================================
  # 3. CREATE INDIVIDUAL SUBPLOT
  # ==========================================
  # Only the left-hand panel carries the legend and the y-axis label
  show_legend = idx == 1 ? :bottomleft : :none
  y_label = idx == 1 ? "Log return " * L"$(\Delta y_\tau)$" * " [%]" : ""

  # Trim the inner margins so the panels sit close together: the leftmost
  # panel drops its right margin, the others shrink their left margin
  margin_left = idx == 1 ? 6mm : 0mm
  margin_right = idx == 1 ? 0mm : 4mm

  # A: 95% predictive band
  p = plot(shock_grid, median_expected_jump,
    ribbon=(median_expected_jump .- pred_lower_95, pred_upper_95 .- median_expected_jump),
    fillalpha=0.2,
    color=:steelblue,
    linewidth=0,
    gridalpha=0.06,
    minorgridalpha=0.025,
    label="",
    title=symbol,
    xlabel="Monetary surprise " * L"$(\Delta i_\tau)$" * " [p.p.]",
    ylabel=y_label,
    ylims=y_limits,
    xlims=(min_shock, max_shock),
    legend=show_legend,
    legendfontsize=8,
    left_margin=margin_left,
    right_margin=margin_right,
    top_margin=2mm,
    bottom_margin=4mm
  )

  # B: Posterior median of the expected jump
  plot!(p, shock_grid, median_expected_jump,
    color=:steelblue,
    linewidth=2,
    label=L"$\hat{\beta}_0 + \hat{\beta}_1 \Delta i_\tau$"
  )

  # C: Scatter of the realized FOMC-window returns
  scatter!(p, observed_shocks, actual_returns,
    color=:crimson,
    markersize=2.5,
    alpha=0.6,
    markerstrokewidth=0,
    label="Observed announcements"
  )

  # Zero reference lines
  hline!(p, [0.0], color=:black, lw=1, linestyle=:dash, label=false)
  vline!(p, [0.0], color=:black, lw=1, linestyle=:dash, label=false)

  push!(subplots, p)
end

# ==========================================
# 4. COMBINE AND SAVE
# ==========================================
println("Generating combined figure...")

mkpath("../figs")

# Arrange the panels horizontally in a single row
n_panels = length(subplots)
final_plot = plot(subplots..., layout=(1, n_panels), size=(400 * n_panels, 350), dpi=300)

output_file = "../figs/stylized_jump_plot.pdf"
savefig(final_plot, output_file)
println("Done. Combined figure saved to $output_file")
