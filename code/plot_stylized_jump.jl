# ==========================================
# plot_stylized_jump_all.jl
# ==========================================
using CSV
using DataFrames
using Statistics
using Distributions
using Plots
using StatsPlots
using LaTeXStrings
using Measures # Necesario para ajustar los márgenes del subplot

# Configuración global
theme(:bright)
symbols = ["SPY", "IWM"]
subplots = []

println("Iniciando generación del gráfico combinado...")

ylims = (-2.1, 2.4)
for (idx, symbol) in enumerate(symbols)
  # ==========================================
  # 1. LOAD DATA & CHAINS
  # ==========================================
  println("Cargando datos para $symbol...")
  master_data = CSV.read("../data/$symbol/$(symbol)_30m.csv", DataFrame)

  # Filtrar solo las ventanas de evento FOMC
  fomc_events = filter(row -> row.is_fomc == true, master_data)
  observed_shocks = fomc_events.target_shock
  actual_returns = fomc_events.adj_log_return

  println("Cargando cadenas MCMC para $symbol...")
  chains_dir = "../mcmc_chains/$symbol/"
  all_files = readdir(chains_dir)
  chain_files = chains_dir .* filter(f -> startswith(f, "chain_") && endswith(f, ".csv"), all_files)

  chains_array = [CSV.read(f, DataFrame) for f in chain_files]
  chain_df = vcat(chains_array...)

  # Extraer parámetros
  beta_0_samples = chain_df.beta_0
  beta_1_samples = chain_df.beta_1
  sigma_eps_samples = chain_df.sigma_eps
  N_samples = length(beta_0_samples)

  # ==========================================
  # 2. COMPUTE PREDICTIVE DISTRIBUTIONS
  # ==========================================
  println("Calculando intervalos predictivos para $symbol...")

  min_shock = minimum(observed_shocks) - 0.02
  max_shock = maximum(observed_shocks) + 0.02
  shock_grid = range(min_shock, max_shock, length=100)

  median_expected_jump = zeros(length(shock_grid))
  pred_lower_95 = zeros(length(shock_grid))
  pred_upper_95 = zeros(length(shock_grid))

  for (i, x) in enumerate(shock_grid)
    # Media esperada para cada muestra MCMC
    expected_means = beta_0_samples .+ beta_1_samples .* x

    # Simulación de las realizaciones añadiendo ruido idiosincrático
    noise = randn(N_samples) .* sigma_eps_samples
    simulated_realizations = expected_means .+ noise

    median_expected_jump[i] = median(expected_means)
    pred_lower_95[i] = quantile(simulated_realizations, 0.025)
    pred_upper_95[i] = quantile(simulated_realizations, 0.975)
  end

  # ==========================================
  # 3. CREATE INDIVIDUAL SUBPLOT
  # ==========================================
  # Configuración condicional para no repetir leyendas ni el eje Y
  show_legend = idx == 1 ? :bottomleft : :none
  y_label = idx == 1 ? "Log-retorno " * L"$(\Delta y_\tau)$" * " [%]" : ""

  # Ajuste fino de márgenes para juntar los paneles:
  # Al SPY (idx=1) le quitamos margen derecho, al IWM (idx=2) le achicamos el izquierdo.
  margen_izq = idx == 1 ? 6mm : 0mm
  margen_der = idx == 1 ? 0mm : 4mm

  # A: Banda predictiva del 95%
  p = plot(shock_grid, median_expected_jump,
    ribbon=(median_expected_jump .- pred_lower_95, pred_upper_95 .- median_expected_jump),
    fillalpha=0.2,
    color=:steelblue,
    linewidth=0,
    gridalpha=0.06,
    minorgridalpha=0.025,
    label="",
    title=symbol,
    xlabel="Sorpresa monetaria " * L"$(\Delta i_\tau)$" * " [p.p.]",
    ylabel=y_label,
    ylims=ylims,
    xlims=(min_shock, max_shock),
    legend=show_legend,
    legendfontsize=8,
    left_margin=margen_izq,
    right_margin=margen_der,
    top_margin=2mm,
    bottom_margin=4mm
  )

  # B: Línea de la media esperada
  plot!(p, shock_grid, median_expected_jump,
    color=:steelblue,
    linewidth=2,
    label=L"$\hat{\beta}_0 + \hat{\beta}_1 \Delta i_\tau$"
  )

  # C: Dispersión de los retornos empíricos
  scatter!(p, observed_shocks, actual_returns,
    color=:crimson,
    markersize=2.5,
    alpha=0.6,
    markerstrokewidth=0,
    label="Anuncios observados"
  )

  # Líneas de referencia (cero)
  hline!(p, [0.0], color=:black, lw=1, linestyle=:dash, label=false)
  vline!(p, [0.0], color=:black, lw=1, linestyle=:dash, label=false)

  push!(subplots, p)
end

# ==========================================
# 4. COMBINE AND SAVE
# ==========================================
println("Generando gráfico combinado...")

# Combinar los gráficos horizontalmente en 1 fila y 2 columnas
final_plot = plot(subplots..., layout=(1, 2), size=(800, 350), dpi=300)

savefig(final_plot, "../figs/stylized_jump_plot.pdf")
println("¡Éxito! El gráfico combinado se guardó en 'stylized_jump_plot_all.pdf'")


