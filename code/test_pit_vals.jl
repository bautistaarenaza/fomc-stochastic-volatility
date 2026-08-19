# ==========================================
# test_pit_vals_all.jl
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
# --- CONFIGURACIÓN PRINCIPAL ---
# ==========================================
symbols = ["SPY", "IWM"]

# Contenedores para almacenar los subpaneles
subplots_acf = []
subplots_acf_sq = []
subplots_cdf = []

# Configuración base de fuente
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
  println("Procesando datos para $symbol...")
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
  println("\nRunning Diebold-Gunther-Tay Independence Tests...")

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
  # 4. PREPARAR CONFIGURACIÓN VISUAL
  # ==========================================
  lags_to_plot = 20
  acf_values = autocor(z_residuals, 1:lags_to_plot)
  acf_values_sq = autocor(z_squared, 1:lags_to_plot)
  conf_bound = 1.96 / sqrt(T_data)

  # Ajuste fino de márgenes y leyendas
  show_legend = idx == 1 ? true : false
  y_label_acf = idx == 1 ? "Autocorrelación" : ""
  y_label_cdf = idx == 1 ? "Probabilidad Acumulada" : "" # Actualizado para la CDF

  margen_izq = idx == 1 ? 6Plots.mm : 2Plots.mm
  margen_der = idx == 1 ? 0Plots.mm : 4Plots.mm

  # ==========================================
  # 5. CREAR SUBPANELES ACF (NIVELES)
  # ==========================================
  p_acf = bar(1:lags_to_plot, acf_values,
    title=symbol,
    xlabel="Lag (intervalos de 30 min)",
    ylabel=y_label_acf,
    label=show_legend ? "Residuos" : "",
    color=:mediumseagreen, linecolor=:darkgreen,
    alpha=0.7, bar_width=0.6,
    legend=show_legend ? :topright : :none,
    left_margin=margen_izq, right_margin=margen_der,
    bottom_margin=6Plots.mm
  )
  hline!(p_acf, [conf_bound], color=:black, linestyle=:dot, lw=2.5, label=show_legend ? "Confianza (95%)" : "")
  hline!(p_acf, [-conf_bound], color=:black, linestyle=:dot, lw=2.5, label="")
  hline!(p_acf, [0.0], color=:black, lw=1, label="")

  max_acf = maximum(abs.(acf_values))
  y_lim_dynamic = max(conf_bound * 2.0, max_acf * 1.2)
  ylims!(p_acf, (-y_lim_dynamic, y_lim_dynamic))
  xticks!(p_acf, 1:2:lags_to_plot)
  push!(subplots_acf, p_acf)

  # ==========================================
  # 6. CREAR SUBPANELES ACF (CUADRADOS)
  # ==========================================
  p_acf_sq = bar(1:lags_to_plot, acf_values_sq,
    title=symbol,
    xlabel="Lag (intervalos de 30 min)",
    ylabel=y_label_acf,
    label=show_legend ? "Residuos al cuadrado" : "",
    color=:orange2, linecolor=:darkorange2,
    alpha=0.7, bar_width=0.6,
    legend=show_legend ? :topright : :none,
    left_margin=margen_izq, right_margin=margen_der,
    bottom_margin=6Plots.mm
  )
  hline!(p_acf_sq, [conf_bound], color=:black, linestyle=:dot, lw=2.5, label=show_legend ? "Confianza (95%)" : "")
  hline!(p_acf_sq, [-conf_bound], color=:black, linestyle=:dot, lw=2.5, label="")
  hline!(p_acf_sq, [0.0], color=:black, lw=1, label="")

  max_acf_sq = maximum(abs.(acf_values_sq))
  y_lim_dynamic_sq = max(conf_bound * 2.0, max_acf_sq * 1.2)
  ylims!(p_acf_sq, (-y_lim_dynamic_sq, y_lim_dynamic_sq))
  xticks!(p_acf_sq, 1:2:lags_to_plot)
  push!(subplots_acf_sq, p_acf_sq)

  # ==========================================
  # 7. CREAR SUBPANELES CDF PIT
  # ==========================================
  # Calcular la distribución empírica acumulada
  sorted_pits = sort(marginal_pit_sequence)
  empirical_cdf = (1:T_data) ./ T_data

  p_cdf = plot(sorted_pits, empirical_cdf,
    title=symbol,
    xlabel=L"$u_t$",
    ylabel=y_label_cdf,
    color=:orangered3,
    linewidth=2,
    label=show_legend ? L"\hat{F}_U(u_t)" * " empírica" : "",
    legend=show_legend ? :topleft : :none, # Ubicado arriba a la izquierda para no pisar la curva
    left_margin=margen_izq, right_margin=margen_der,
    bottom_margin=5Plots.mm
  )

  # Añadir la CDF teórica de una Uniforme(0,1) que es y = x
  plot!(p_cdf, [0.0, 1.0], [0.0, 1.0],
    color=:black, alpha=0.5, ls=:dot, lw=2,
    label=show_legend ? L"F_U(u_t)" * " teórica " * L"\mathcal{U}(0,1)" : "")

  ylims!(p_cdf, (0.0, 1.0))
  xlims!(p_cdf, (0.0, 1.0))
  push!(subplots_cdf, p_cdf)
end

# ==========================================
# 8. COMBINAR Y GUARDAR LOS GRÁFICOS
# ==========================================
println("\nGenerando gráficos combinados...")

# Desactivar la grilla por defecto para los gráficos finales
default(grid=false, framestyle=:axes, dpi=300)

final_acf = plot(subplots_acf..., layout=(1, 2), size=(1000, 350))
savefig(final_acf, "../figs/combined_acf_res.pdf")

final_acf_sq = plot(subplots_acf_sq..., layout=(1, 2), size=(1000, 350))
savefig(final_acf_sq, "../figs/combined_acf_res2.pdf")

# Ahora guardamos el gráfico combinado de las CDF
final_cdf = plot(subplots_cdf..., layout=(1, 2), size=(1000, 300))
savefig(final_cdf, "../figs/combined_pit_cdf.pdf")

println("¡Éxito! Gráficos guardados en el directorio '../figs/' con prefijo 'combined_'.")

