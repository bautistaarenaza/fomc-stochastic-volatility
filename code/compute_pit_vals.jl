# ==========================================
# compute_pit_vals.jl
# ==========================================
include("MacroFinanceModel.jl")
using .MacroFinanceModel

using CSV
using DataFrames
using Statistics
using SpecialFunctions

# ==========================================
# --- CONFIGURACIÓN PRINCIPAL ---
# ==========================================
symbol = "SPY"

# Raw CDF function to avoid Distributions.jl allocations
function normcdf(μ, σ, x)
  return 0.5 * (1.0 + erf((x - μ) / (σ * sqrt(2.0))))
end

# ==========================================
# 1. THE SPECIALIZED PIT PARTICLE FILTER
# ==========================================
function systematic_resample(weights::Vector{Float64}, N::Int)
  indices = zeros(Int, N)
  u_1 = rand() / N
  c = weights[1]
  i = 1

  @inbounds for j in 1:N
    u_j = u_1 + (j - 1) / N
    while u_j > c && i < N
      i += 1
      c += weights[i]
    end
    indices[j] = i
  end

  return indices
end

function run_particle_filter_pit(params::ModelParams, data::DataFrame, N_particles::Int)
  T = nrow(data)
  pit_values = zeros(Float64, T)

  x_prev = fill(params.θ, N_particles)
  x_next = zeros(Float64, N_particles)
  x_prev_prev = fill(params.θ, N_particles)
  weights = zeros(Float64, N_particles)
  cdf_vals = zeros(Float64, N_particles)

  y_data = Vector{Float64}(data.adj_log_return)
  shock_data = Vector{Float64}(data.target_shock)
  dt_data = Vector{Float64}(data.dt)
  is_fomc_data = Vector{Bool}(data.is_fomc)

  inv_sqrt_2pi = 1.0 / sqrt(2π)
  log_2pi_half = -0.5 * log(2π)

  # Trackers for the Jitter's look-back requirement
  y_prev = 0.0
  shock_prev = 0.0
  dt_prev = dt_data[1]
  is_fomc_prev = false

  @inbounds for t in 1:T
    y_t = y_data[t]
    shock = shock_data[t]
    dt = dt_data[t]
    is_fomc = is_fomc_data[t]

    Threads.@threads for i in 1:N_particles
      x_safe = clamp(x_prev[i], -20.0, 20.0)
      v_eval = exp(x_safe)
      drift_penalty = 0.5 * (v_eval / 100.0)

      continuous_mean = (params.μ - drift_penalty) * dt
      continuous_var = v_eval * dt

      if !is_fomc
        std_dev = sqrt(continuous_var)
        cdf_vals[i] = normcdf(continuous_mean, std_dev, y_t)
        weights[i] = (inv_sqrt_2pi / std_dev) * exp(-0.5 * ((y_t - continuous_mean) / std_dev)^2)
      else
        jump_mean = params.β_0 + params.β_1 * shock
        total_mean = continuous_mean + jump_mean
        total_std = sqrt(continuous_var + params.σ_ε^2)
        cdf_vals[i] = normcdf(total_mean, total_std, y_t)
        weights[i] = (inv_sqrt_2pi / total_std) * exp(-0.5 * ((y_t - total_mean) / total_std)^2)
      end
    end

    pit_values[t] = mean(cdf_vals)

    mean_weight = mean(weights)
    if mean_weight > 0 && !isnan(mean_weight)
      weights .= weights ./ sum(weights)
    else
      weights .= 1.0 / N_particles
    end

    ess_t = 1.0 / sum(weights .^ 2)

    indices = systematic_resample(weights, N_particles)
    x_resampled = x_prev[indices]
    x_prev_prev_resampled = x_prev_prev[indices]

    # -------------------------------------------------------------------
    # FULL POSTERIOR JITTER (WITH LEVERAGE & LOOK-BACK FIX)
    # -------------------------------------------------------------------
    if ess_t < (N_particles / 2.0)
      emp_std = std(x_resampled)
      sigma_jitter = max(emp_std * 0.20, 0.05)

      Threads.@threads for i in 1:N_particles
        x_curr = x_resampled[i]
        x_prop = x_curr + randn() * sigma_jitter
        prev_state = x_prev_prev_resampled[i]

        if t > 1
          v_prev_eval = exp(prev_state)
          drift_pen_prev = 0.5 * (v_prev_eval / 100.0)
          cont_mean_prev = (params.μ - drift_pen_prev) * dt_prev

          if !is_fomc_prev
            obs_var_prev = v_prev_eval * dt_prev
            eta_prev = y_prev - cont_mean_prev
            jump_comp = 0.0
          else
            jump_mean_prev = params.β_0 + params.β_1 * shock_prev
            obs_var_prev = v_prev_eval * dt_prev + params.σ_ε^2
            eta_prev = y_prev - cont_mean_prev - jump_mean_prev
            jump_comp = params.α_0 + params.α_1 * abs(shock_prev)
          end

          cov_Wv_eta_prev = sqrt(v_prev_eval) * params.ρ * dt_prev
          cond_mean_Wv_prev = (cov_Wv_eta_prev / obs_var_prev) * eta_prev
          cond_var_Wv_prev = dt_prev - (cov_Wv_eta_prev^2 / obs_var_prev)

          drift_base = params.κ * (params.θ - prev_state) * dt_prev
          expected_mean = prev_state + drift_base + jump_comp + params.σ_v * cond_mean_Wv_prev
          transition_std = params.σ_v * sqrt(max(cond_var_Wv_prev, 0.0))
        else
          drift_base = params.κ * (params.θ - prev_state) * dt
          expected_mean = prev_state + drift_base
          transition_std = params.σ_v * sqrt(dt)
        end

        log_trans_prop = log_2pi_half - log(transition_std) - 0.5 * ((x_prop - expected_mean) / transition_std)^2
        log_trans_curr = log_2pi_half - log(transition_std) - 0.5 * ((x_curr - expected_mean) / transition_std)^2

        v_prop = exp(x_prop)
        cont_mean_prop = (params.μ - 0.5 * (v_prop / 100.0)) * dt
        cont_var_prop = v_prop * dt

        v_curr = exp(x_curr)
        cont_mean_curr = (params.μ - 0.5 * (v_curr / 100.0)) * dt
        cont_var_curr = v_curr * dt

        if !is_fomc
          std_prop = sqrt(cont_var_prop)
          std_curr = sqrt(cont_var_curr)
          log_obs_prop = log_2pi_half - log(std_prop) - 0.5 * ((y_t - cont_mean_prop) / std_prop)^2
          log_obs_curr = log_2pi_half - log(std_curr) - 0.5 * ((y_t - cont_mean_curr) / std_curr)^2
        else
          jump_mean = params.β_0 + params.β_1 * shock
          std_prop = sqrt(cont_var_prop + params.σ_ε^2)
          std_curr = sqrt(cont_var_curr + params.σ_ε^2)
          log_obs_prop = log_2pi_half - log(std_prop) - 0.5 * ((y_t - (cont_mean_prop + jump_mean)) / std_prop)^2
          log_obs_curr = log_2pi_half - log(std_curr) - 0.5 * ((y_t - (cont_mean_curr + jump_mean)) / std_curr)^2
        end

        log_alpha = (log_trans_prop + log_obs_prop) - (log_trans_curr + log_obs_curr)

        if log(rand()) < log_alpha
          x_resampled[i] = x_prop
        end
      end
    end
    # -------------------------------------------------------------------

    Threads.@threads for i in 1:N_particles
      x_curr = x_resampled[i]
      v_eval = exp(x_curr)

      drift_penalty = 0.5 * (v_eval / 100.0)
      cont_mean = (params.μ - drift_penalty) * dt

      if !is_fomc
        obs_var = v_eval * dt
        eta = y_t - cont_mean
      else
        jump_mean = params.β_0 + params.β_1 * shock
        obs_var = v_eval * dt + params.σ_ε^2
        eta = y_t - cont_mean - jump_mean
      end

      cov_Wv_eta = sqrt(v_eval) * params.ρ * dt
      cond_mean_Wv = (cov_Wv_eta / obs_var) * eta
      cond_var_Wv = dt - (cov_Wv_eta^2 / obs_var)

      dW_V = cond_mean_Wv + sqrt(max(cond_var_Wv, 0.0)) * randn()

      drift_x = params.κ * (params.θ - x_curr) * dt
      x_new = x_curr + drift_x + params.σ_v * dW_V

      if is_fomc
        Z_x = params.α_0 + params.α_1 * abs(shock)
        x_new += Z_x
      end

      x_next[i] = clamp(x_new, -20.0, 20.0)
    end

    x_prev_prev .= x_prev
    x_prev .= x_next

    # Update trackers for the next step's Jitter look-back
    y_prev = y_t
    shock_prev = shock
    dt_prev = dt
    is_fomc_prev = is_fomc
  end

  return pit_values
end

# ==========================================
# 2. LOAD DATA & CHAINS
# ==========================================
println("Loading data for $symbol...")
filename = "../data/$symbol/$(symbol)_30m.csv"
master_data = CSV.read(filename, DataFrame)

chains_dir = "../mcmc_chains/$symbol/"
all_files = readdir(chains_dir)
chain_files = chains_dir .* filter(f -> startswith(f, "chain_") && endswith(f, ".csv"), all_files)

if isempty(chain_files)
  println("ERROR: No chain files found.")
  exit()
end

println("Aggregating $(length(chain_files)) chain files...")
chains_array = [CSV.read(f, DataFrame) for f in chain_files]
master_chain_df = vcat(chains_array...)

core_params = ["kappa", "theta", "sigma_v", "mu", "alpha_0", "alpha_1", "beta_0", "beta_1", "sigma_eps", "rho"]

# ==========================================
# 3. FULL BAYESIAN MARGINALIZATION 
# ==========================================
T_data = nrow(master_data)
marginal_pit_sequence = zeros(Float64, T_data)

N_posterior_samples = 100
step_size = max(1, nrow(master_chain_df) ÷ N_posterior_samples)
sampled_indices = 1:step_size:nrow(master_chain_df)
sampled_indices = sampled_indices[1:min(N_posterior_samples, length(sampled_indices))]

println("\n🚀 Starting Full Bayesian Marginalization...")
println("Thinning chain: Evaluating $(length(sampled_indices)) posterior samples...")

N_eval_particles = 2^14

for (idx, row_idx) in enumerate(sampled_indices)
  theta_s = [master_chain_df[row_idx, p] for p in core_params]
  params_s = ModelParams(theta_s...)

  pit_s = run_particle_filter_pit(params_s, master_data, N_eval_particles)
  marginal_pit_sequence .+= pit_s

  print("\r  Processed sample $idx / $(length(sampled_indices))")
end
println("\n")

marginal_pit_sequence ./= length(sampled_indices)

# ==========================================
# 4. SAVE PIT SEQUENCE
# ==========================================
println("Saving PIT values to CSV...")
mkpath("../pit_vals")
output_file = "../pit_vals/$(symbol)_pit_sequence.csv"

# Se extrae la columna ny_time directamente del dataset original
out_df = DataFrame(
  ny_time=master_data.ny_time,
  pit_value=marginal_pit_sequence
)

CSV.write(output_file, out_df)
println("Successfully saved marginalized PIT sequence to: $output_file")

