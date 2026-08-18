# ==========================================
# run_pmcmc.jl
# ==========================================
include("MacroFinanceModel.jl")
using .MacroFinanceModel

using DataFrames
using CSV
using Distributions
using Random
using Statistics
using LinearAlgebra
using Dates

# ==========================================
# --- MAIN CONFIGURATION ---
# ==========================================
symbol = "SPY"

# ==========================================
# 1. LOAD DATA
# ==========================================
println("Loading data for $symbol...")
filename = "../data/$symbol/$(symbol)_30m.csv"

master_data = CSV.read(filename, DataFrame)

filter!(row -> !ismissing(row.adj_log_return) && !isnan(row.adj_log_return), master_data)

if eltype(master_data.ny_time) == String
  master_data.ny_time = DateTime.(master_data.ny_time, dateformat"yyyy-mm-dd HH:MM:SS")
end

start_date = DateTime(2016, 1, 1, 0, 0, 0)
end_date = DateTime(2026, 3, 31, 23, 59, 59)

filter!(row -> start_date <= row.ny_time <= end_date, master_data)

println("Dataset loaded: $(nrow(master_data)) rows.")

# ==========================================
# 2. DEFINE THE LOG-PRIOR
# ==========================================
function generate_asset_prior(master_data::DataFrame)
  println("Generating data-driven prior for new asset...")

  clean_bars = filter(row -> !row.is_fomc, master_data)
  fomc_bars = filter(row -> row.is_fomc, master_data)

  # DYNAMIC ANNUALIZATION
  # Como 'dt' ya es la fracción de año por ventana, su inversa es el número exacto de ventanas por año.
  median_dt = median(clean_bars.dt)
  ann_factor = 1.0 / median_dt

  # Solo para métricas en pantalla
  median_dt_mins = median(clean_bars.dt_minutes)
  implied_bars_per_day = ann_factor / 252.0

  println("  [System] Detected $(median_dt_mins)-minute data grid.")
  println("  [System] Implied bars per day: $(round(implied_bars_per_day, digits=2))")
  println("  [System] Using annualization factor: $(round(ann_factor, digits=2))")

  est_var = var(clean_bars.adj_log_return) * ann_factor
  est_theta = log(est_var)

  raw_mean = mean(clean_bars.adj_log_return) * ann_factor
  est_mu = raw_mean + 0.5 * (est_var / 100.0)

  x = fomc_bars.target_shock
  y = fomc_bars.adj_log_return

  est_beta_1 = cov(x, y) / var(x)
  est_beta_0 = mean(y) - est_beta_1 * mean(x)

  residuals = y .- (est_beta_0 .+ est_beta_1 .* x)
  est_sigma_eps = std(residuals)

  println("--- Extracted Prior Means ---")
  println("  Drift (μ):        $(round(est_mu, digits=3))")
  println("  Log-Var (θ):      $(round(est_theta, digits=3))")
  println("  Jump Int (β_0):   $(round(est_beta_0, digits=3))")
  println("  Jump Slope (β_1): $(round(est_beta_1, digits=3))")
  println("-----------------------------")

  return function dynamic_log_prior(θ_array)
    κ, θ, σ_v, μ, α_0, α_1, β_0, β_1, σ_ε, ρ = θ_array

    # Bound safety checks including the new correlation parameter
    if κ <= 0.0 || σ_v <= 0.0 || σ_ε <= 0.0 || ρ <= -0.99 || ρ >= 0.99
      return -Inf
    end

    lp = 0.0

    lp += logpdf(LogNormal(2.0, 1.0), κ)
    lp += logpdf(LogNormal(2.0, 1.0), σ_v)
    lp += logpdf(InverseGamma(2.0, est_sigma_eps * 3.0), σ_ε)

    # Prior for Leverage Effect: Centered around -0.5, allowing broad exploration
    lp += logpdf(Normal(-0.5, 0.5), ρ)

    lp += logpdf(Normal(est_theta, 3.0), θ)
    lp += logpdf(Normal(est_mu, 3.0), μ)

    lp += logpdf(Normal(est_beta_0, 2.0), β_0)
    lp += logpdf(Normal(est_beta_1, 3.0), β_1)

    lp += logpdf(Normal(0.0, 2.0), α_0)
    lp += logpdf(Normal(0.5, 2.0), α_1)

    return lp
  end
end

# ==========================================
# 3. PMCMC SETUP (Wrapped in a function)
# ==========================================
function run_sampler(master_data::DataFrame, symbol::String)
  N_particles = 2^14
  burn_in = 4000
  valid_chain_length = 5000
  M_iterations = burn_in + valid_chain_length
  d = 10

  base_θ = if symbol == "SPY"
    [69.1320, 4.0525, 13.9050, 10.7667, 1.4718, 0.8651, 0.0254, -0.6774, 0.3683, -0.2962]
  elseif symbol == "IWM"
    [73.9841, 5.0131, 11.1485, 2.2969, 1.2881, 0.7779, 0.0849, -2.2967, 0.6064, -0.2464]
  elseif symbol == "SHY"
    [76.722, -0.516, 13.578, 0.055, 0.972, 2.242, -0.011, -0.451, 0.085, 0.005]
  end
  param_names = ["kappa", "theta", "sigma_v", "mu", "alpha_0", "alpha_1", "beta_0", "beta_1", "sigma_eps", "rho"]

  out_dir = "mcmc_chains/$symbol"
  mkpath(out_dir)
  checkpoint_file = "$out_dir/checkpoint_N$(N_particles).csv"

  chain = zeros(M_iterations, d)
  log_posterior_trace = zeros(M_iterations)
  valid_accepted_steps = 0
  local_accepted = 0

  initial_std = 0.05 * ones(d)
  initial_cov = diagm(initial_std .^ 2)
  prop_cov = copy(initial_cov)
  beta = 0.05
  global_scale = (2.38^2) / d

  log_prior = generate_asset_prior(master_data)

  # ---------------------------------------------------------
  # THE RESUME LOGIC
  # ---------------------------------------------------------
  start_iter = 1

  if isfile(checkpoint_file)
    println("\n[PMCMC] 🟢 Checkpoint detected! Restoring state from $checkpoint_file...")
    df_cp = CSV.read(checkpoint_file, DataFrame)
    start_iter = nrow(df_cp) + 1

    if start_iter > M_iterations
      println("Checkpoint already contains $M_iterations steps. Bypassing sampling and generating final files...")
    end

    for r in 1:min(start_iter - 1, M_iterations)
      chain[r, :] = Vector{Float64}(df_cp[r, 1:d])
      log_posterior_trace[r] = df_cp.log_posterior[r]
    end

    current_θ = Vector{Float64}(df_cp[start_iter-1, 1:d])
    current_posterior = df_cp.log_posterior[start_iter-1]
    global_scale = df_cp.global_scale[start_iter-1]

    if start_iter > burn_in + 1
      for j in (burn_in+2):(start_iter-1)
        if chain[j, :] != chain[j-1, :]
          valid_accepted_steps += 1
        end
      end
    end
    println("Resuming at iteration $start_iter. Restored Log-Posterior: $(round(current_posterior, digits=2))")
  else
    println("\n[PMCMC] Initializing Fresh Sampler for $symbol...")
    println("Base Coordinates: ", round.(base_θ, digits=3))

    # --- INJECT NOISE HERE ---
    noise_fraction = 0.05 # 5% relative noise
    valid_start = false
    current_θ = copy(base_θ)
    current_posterior = -Inf

    println("Hunting for a valid 5% overdispersed starting coordinate...")
    while !valid_start
      proposed_noisy_θ = base_θ .+ noise_fraction * randn(d) .* abs.(base_θ)
      proposed_prior = log_prior(proposed_noisy_θ)

      if proposed_prior > -Inf
        # It passed the prior bounds, now test if the filter survives it
        p_struct = ModelParams(proposed_noisy_θ...)
        proposed_ll = run_particle_filter(p_struct, master_data, N_particles)

        if proposed_ll > -Inf
          current_θ = proposed_noisy_θ
          current_posterior = proposed_ll + proposed_prior
          valid_start = true
        end
      end
    end

    println("Noisy Starting Coordinates: ", round.(current_θ, digits=3))

    params_struct = ModelParams(current_θ...)
    current_ll = run_particle_filter(params_struct, master_data, N_particles, diagnostic=true)
    current_posterior = current_ll + log_prior(current_θ)

    println("Initial Log-Posterior: ", round(current_posterior, digits=2))
  end

  println("\n[PMCMC] Starting Adaptive Covariance M-H Loop ($M_iterations total iterations)...\n")
  start_time = time()

  for i in start_iter:M_iterations

    if i > 1000
      start_idx = max(1, i - 1000)
      empirical_cov = cov(chain[start_idx:(i-1), :])
      mixed_cov = (1.0 - beta) * empirical_cov + beta * initial_cov
      prop_cov = global_scale * mixed_cov
    elseif i > 500
      start_idx = max(1, i - 500)
      empirical_var = vec(var(chain[start_idx:(i-1), :], dims=1))
      mixed_cov = (1.0 - beta) * diagm(empirical_var) + beta * initial_cov
      prop_cov = global_scale * mixed_cov
    end

    jitter = 1e-6 * I
    L_matrix = cholesky(Symmetric(prop_cov + jitter)).L
    proposed_θ = current_θ .+ L_matrix * randn(d)

    proposed_prior = log_prior(proposed_θ)

    if proposed_prior == -Inf
      proposed_posterior = -Inf
    else
      p_struct = ModelParams(proposed_θ...)
      proposed_ll = run_particle_filter(p_struct, master_data, N_particles)
      proposed_posterior = proposed_ll + proposed_prior
    end

    log_alpha = proposed_posterior - current_posterior

    if log(rand()) < log_alpha
      current_θ = copy(proposed_θ)
      current_posterior = proposed_posterior

      local_accepted += 1
      if i > burn_in
        valid_accepted_steps += 1
      end
    end

    chain[i, :] = current_θ
    log_posterior_trace[i] = current_posterior

    # ---------------------------------------------------------
    # THE CHECKPOINT SAVER
    # ---------------------------------------------------------
    if i % 250 == 0
      elapsed_secs = time() - start_time
      e_mins = floor(Int, elapsed_secs / 60)
      e_secs = round(Int, elapsed_secs % 60)
      time_str = "$(e_mins)m $(e_secs)s"

      # Calculate Argentina Time (UTC-3)
      arg_time_str = Dates.format(Dates.now(Dates.UTC) - Dates.Hour(3), "yyyy-mm-dd HH:MM:SS")

      if i <= burn_in
        local_acc_rate = local_accepted / 250.0

        if local_acc_rate < 0.10
          global_scale *= 0.90
        elseif local_acc_rate > 0.20
          global_scale *= 1.10
        end
        global_scale = clamp(global_scale, 1e-5, 1.0)

        phase_str = i <= 1000 ? "Phase 1" : "Phase 2"
        println("BURN-IN $i / $burn_in | $phase_str | Acc: $(round(local_acc_rate*100, digits=1))% | Scale: $(round(global_scale, digits=5)) | LP: $(round(current_posterior, digits=2)) | Elapsed: $time_str | Arg Time: $arg_time_str")
      else
        valid_steps_taken = i - burn_in
        overall_acc = round((valid_accepted_steps / valid_steps_taken) * 100, digits=2)
        println("SAMPLING $valid_steps_taken / $valid_chain_length | Overall Acc: $overall_acc% | LP: $(round(current_posterior, digits=2)) | Elapsed: $time_str | Arg Time: $arg_time_str")
      end

      println(" - current θ = ", round.(current_θ, digits=3))
      println(" - mean θ    = ", round.(mean(chain[max(1, i - 249):i, :], dims=1), digits=3))
      println(" - std θ     = ", round.(std(chain[max(1, i - 249):i, :], dims=1), digits=3))
      println()

      cp_df = DataFrame(chain[1:i, :], param_names)
      cp_df[!, :log_posterior] = log_posterior_trace[1:i]
      cp_df[!, :global_scale] = fill(global_scale, i)
      CSV.write(checkpoint_file, cp_df)

      local_accepted = 0
    end
  end

  println("\n[PMCMC] Sampling Complete for $(symbol)!")
  final_acc_rate = valid_accepted_steps / valid_chain_length
  println("Final Valid Acceptance Rate: ", round(final_acc_rate * 100, digits=2), "%")

  valid_chain = chain[(burn_in+1):end, :]
  valid_log_post = log_posterior_trace[(burn_in+1):end]

  results_df = DataFrame(
    Parameter=param_names,
    Posterior_Mean=mean(valid_chain, dims=1)[1, :],
    Posterior_StdErr=std(valid_chain, dims=1)[1, :]
  )

  display(results_df)

  file_idx = 1
  base_filename = "$out_dir/chain_N$(N_particles)_L$(valid_chain_length)"
  final_filename = "$(base_filename)_$(file_idx).csv"

  while isfile(final_filename)
    file_idx += 1
    final_filename = "$(base_filename)_$(file_idx).csv"
  end

  save_df = DataFrame(valid_chain, param_names)
  save_df[!, :log_posterior] = valid_log_post
  CSV.write(final_filename, save_df)
  println("\n✅ Saved final valid chain for $symbol to $(final_filename)")

  # ---------------------------------------------------------
  # CHECKPOINT CLEANUP
  # ---------------------------------------------------------
  if isfile(checkpoint_file)
    rm(checkpoint_file, force=true)
    println("🗑️  Cleaned up checkpoint file: $(checkpoint_file)")
  end
end

# ==========================================
# EXECUTE THE FUNCTION
# ==========================================
run_sampler(master_data, symbol)

