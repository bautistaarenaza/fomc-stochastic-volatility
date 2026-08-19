# ==========================================
# test_pit_vals.jl
# ==========================================
using CSV
using DataFrames
using Distributions
using Statistics
using Plots
using StatsPlots
using HypothesisTests
using StatsBase
using LaTeXStrings
using Measures

# ==========================================
# --- MAIN CONFIGURATION ---
# ==========================================
symbols = ["SPY", "IWM"]

# Containers for the per-symbol subpanels
subplots_acf = []
subplots_acf_sq = []
subplots_cdf = []

# Global plot defaults. These must be set before any plot object is built,
# since Plots.jl resolves attributes at creation time.
default(grid=false, framestyle=:axes, dpi=300)
scalefontsizes(1.2)

for (idx, symbol) in enumerate(symbols)
  # ==========================================
  # 1. LOAD SAVED PIT SEQUENCE
  # ==========================================
  input_file = "../pit_vals/$(symbol)_pit_sequence.csv"
  if !isfile(input_file)
    println("ERROR: Could not find $input_file. Run the calculation script first.")
    continue
  end

  println("\n==================================================")
  println("Processing data for $symbol...")
  println("==================================================")

  pit_df = CSV.read(input_file, DataFrame)
  marginal_pit_sequence = pit_df.pit_value
  T_data = length(marginal_pit_sequence)

  # ==========================================
  # 2. STATISTICAL UNIFORMITY TESTING (K-S Test)
  # ==========================================
  println("\nRunning Kolmogorov-Smirnov Test for Uniformity...")
  ks_test = ExactOneSampleKSTest(marginal_pit_sequence, Uniform(0.0, 1.0))
  ks_stat = ks_test.δ
  ks_pvalue = pvalue(ks_test)

  println("Kolmogorov-Smirnov Test Results:")
  println("  Max Distance (D): $(round(ks_stat, digits=4))")
  println("  P-Value:          $(round(ks_pvalue, digits=6))")
  if ks_pvalue > 0.05
    println("  Conclusion: PASS. We fail to reject the null hypothesis.")
  else
    println("  Conclusion: FAIL. We reject the null hypothesis.")
  end

  # ==========================================
  # 3. DGT INDEPENDENCE TESTING (Ljung-Box)
  # ==========================================
  # Diebold-Gunther-Tay: map the PITs through the inverse normal CDF and test
  # the resulting z-scores for serial correlation. Levels probe the mean
  # equation, squares probe the variance equation.
  println("\nRunning Diebold-Gunther-Tay Independence Tests...")

  # Clamp away from the open endpoints so the inverse CDF stays finite
  safe_pits = clamp.(marginal_pit_sequence, 1e-7, 1.0 - 1e-7)
  z_residuals = quantile.(Normal(0, 1), safe_pits)
  z_squared = z_residuals .^ 2
  lags_to_test = 20

  lb_levels = LjungBoxTest(z_residuals, lags_to_test)
  pval_levels = pvalue(lb_levels)

  lb_squares = LjungBoxTest(z_squared, lags_to_test)
  pval_squares = pvalue(lb_squares)

  println("Ljung-Box Q-Test (Lags = $lags_to_test):")
  println("  Levels (Mean Eq) P-Value:   $(round(pval_levels, digits=6))")
  println("  Squares (Var Eq) P-Value:   $(round(pval_squares, digits=6))")

  # ==========================================
  # 4. VISUAL CONFIGURATION
  # ==========================================
  lags_to_plot = 20
  acf_values = autocor(z_residuals, 1:lags_to_plot)
  acf_values_sq = autocor(z_squared, 1:lags_to_plot)
  conf_bound = 1.96 / sqrt(T_data)

  # Only the left-hand panel carries the y-label and the legend
  show_legend = idx == 1 ? true : false
  y_label_acf = idx == 1 ? "Autocorrelation" : ""
  y_label_cdf = idx == 1 ? "Cumulative Probability" : ""

  margin_left = idx == 1 ? 6Plots.mm : 2Plots.mm
  margin_right = idx == 1 ? 0Plots.mm : 4Plots.mm

  # ==========================================
  # 5. ACF SUBPANELS (LEVELS)
  # ==========================================
  p_acf = bar(1:lags_to_plot, acf_values,
    title=symbol,
    xlabel="Lag (30-minute intervals)",
    ylabel=y_label_acf,
    label=show_legend ? "Residuals" : "",
    color=:mediumseagreen, linecolor=:darkgreen,
    alpha=0.7, bar_width=0.6,
    legend=show_legend ? :topright : :none,
    left_margin=margin_left, right_margin=margin_right,
    bottom_margin=6Plots.mm
  )
  hline!(p_acf, [conf_bound], color=:black, linestyle=:dot, lw=2.5, label=show_legend ? "95% confidence band" : "")
  hline!(p_acf, [-conf_bound], color=:black, linestyle=:dot, lw=2.5, label="")
  hline!(p_acf, [0.0], color=:black, lw=1, label="")

  # Scale the y-axis so the confidence band is always visible, even when the
  # autocorrelations themselves are tiny
  max_acf = maximum(abs.(acf_values))
  y_lim_dynamic = max(conf_bound * 2.0, max_acf * 1.2)
  ylims!(p_acf, (-y_lim_dynamic, y_lim_dynamic))
  xticks!(p_acf, 1:2:lags_to_plot)
  push!(subplots_acf, p_acf)

  # ==========================================
  # 6. ACF SUBPANELS (SQUARES)
  # ==========================================
  p_acf_sq = bar(1:lags_to_plot, acf_values_sq,
    title=symbol,
    xlabel="Lag (30-minute intervals)",
    ylabel=y_label_acf,
    label=show_legend ? "Squared residuals" : "",
    color=:orange2, linecolor=:darkorange2,
    alpha=0.7, bar_width=0.6,
    legend=show_legend ? :topright : :none,
    left_margin=margin_left, right_margin=margin_right,
    bottom_margin=6Plots.mm
  )
  hline!(p_acf_sq, [conf_bound], color=:black, linestyle=:dot, lw=2.5, label=show_legend ? "95% confidence band" : "")
  hline!(p_acf_sq, [-conf_bound], color=:black, linestyle=:dot, lw=2.5, label="")
  hline!(p_acf_sq, [0.0], color=:black, lw=1, label="")

  max_acf_sq = maximum(abs.(acf_values_sq))
  y_lim_dynamic_sq = max(conf_bound * 2.0, max_acf_sq * 1.2)
  ylims!(p_acf_sq, (-y_lim_dynamic_sq, y_lim_dynamic_sq))
  xticks!(p_acf_sq, 1:2:lags_to_plot)
  push!(subplots_acf_sq, p_acf_sq)

  # ==========================================
  # 7. PIT CDF SUBPANELS
  # ==========================================
  # Empirical CDF of the PIT sequence, compared against the U(0,1) diagonal
  sorted_pits = sort(marginal_pit_sequence)
  empirical_cdf = (1:T_data) ./ T_data

  p_cdf = plot(sorted_pits, empirical_cdf,
    title=symbol,
    xlabel=L"$u_t$",
    ylabel=y_label_cdf,
    color=:orangered3,
    linewidth=2,
    label=show_legend ? L"\hat{F}_U(u_t)" * " empirical" : "",
    legend=show_legend ? :topleft : :none, # Top-left keeps the legend clear of the curve
    left_margin=margin_left, right_margin=margin_right,
    bottom_margin=5Plots.mm
  )

  # Theoretical CDF of a Uniform(0,1), i.e. the line y = x
  plot!(p_cdf, [0.0, 1.0], [0.0, 1.0],
    color=:black, alpha=0.5, ls=:dot, lw=2,
    label=show_legend ? L"F_U(u_t)" * " theoretical " * L"\mathcal{U}(0,1)" : "")

  ylims!(p_cdf, (0.0, 1.0))
  xlims!(p_cdf, (0.0, 1.0))
  push!(subplots_cdf, p_cdf)
end

# ==========================================
# 8. COMBINE AND SAVE THE FIGURES
# ==========================================
println("\nGenerating combined figures...")

mkpath("../figs")

n_panels = length(subplots_acf)
if n_panels == 0
  error("No PIT sequences were loaded. Run compute_pit_vals.jl first.")
end

final_acf = plot(subplots_acf..., layout=(1, n_panels), size=(500 * n_panels, 350))
savefig(final_acf, "../figs/combined_acf_res.pdf")

final_acf_sq = plot(subplots_acf_sq..., layout=(1, n_panels), size=(500 * n_panels, 350))
savefig(final_acf_sq, "../figs/combined_acf_res2.pdf")

final_cdf = plot(subplots_cdf..., layout=(1, n_panels), size=(500 * n_panels, 300))
savefig(final_cdf, "../figs/combined_pit_cdf.pdf")

# scalefontsizes is cumulative within a session; reset so repeated runs in the
# same REPL do not keep enlarging the fonts
Plots.resetfontsizes()

println("Done. Figures saved to '../figs/' with the 'combined_' prefix.")
