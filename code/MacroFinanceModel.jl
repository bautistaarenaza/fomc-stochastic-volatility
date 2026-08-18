module MacroFinanceModel

using DataFrames
using LinearAlgebra
using Statistics
using Random

export ModelParams, run_particle_filter

struct ModelParams
  κ::Float64
  θ::Float64
  σ_v::Float64
  μ::Float64
  α_0::Float64
  α_1::Float64
  β_0::Float64
  β_1::Float64
  σ_ε::Float64
  ρ::Float64
end

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

function run_particle_filter(params::ModelParams, data::DataFrame, N_particles::Int; diagnostic::Bool=false)
  T = nrow(data)
  log_likelihood = 0.0

  total_ess = 0.0
  min_ess = Float64(N_particles)

  x_prev = fill(params.θ, N_particles)
  x_next = zeros(Float64, N_particles)
  x_prev_prev = fill(params.θ, N_particles)
  weights = zeros(Float64, N_particles)

  y_data = Vector{Float64}(data.adj_log_return)
  shock_data = Vector{Float64}(data.target_shock)
  dt_data = Vector{Float64}(data.dt)
  is_fomc_data = Vector{Bool}(data.is_fomc)

  # Trackers for the jitter step's look-back requirement
  y_prev = 0.0
  shock_prev = 0.0
  dt_prev = dt_data[1]
  is_fomc_prev = false

  # Pre-calculate constants
  inv_sqrt_2pi = 1.0 / sqrt(2π)
  log_2pi_half = -0.5 * log(2π)

  @inbounds for t in 1:T
    y_t = y_data[t]
    shock = shock_data[t]
    dt = dt_data[t]
    is_fomc = is_fomc_data[t]

    # -------------------------------------------------------------------
    # Evaluates p(y_t | x_{t-1}) for each particle
    # -------------------------------------------------------------------
    Threads.@threads for i in 1:N_particles
      x_safe = clamp(x_prev[i], -20.0, 20.0)
      v_eval = exp(x_safe)
      drift_penalty = 0.5 * (v_eval / 100.0)      # v_eval/100 because returns are in %s

      continuous_mean = (params.μ - drift_penalty) * dt
      continuous_var = v_eval * dt

      if !is_fomc
        std_dev = sqrt(continuous_var)
        weights[i] = (inv_sqrt_2pi / std_dev) * exp(-0.5 * ((y_t - continuous_mean) / std_dev)^2)
      else
        jump_mean = params.β_0 + params.β_1 * shock
        total_mean = continuous_mean + jump_mean
        total_std = sqrt(continuous_var + params.σ_ε^2)
        weights[i] = (inv_sqrt_2pi / total_std) * exp(-0.5 * ((y_t - total_mean) / total_std)^2)
      end
    end

    # -------------------------------------------------------------------
    # Resamples particles according to the p(y_t | x_{t-1}) weights
    # -------------------------------------------------------------------
    mean_weight = mean(weights)
    if mean_weight > 0 && !isnan(mean_weight)
      log_likelihood += log(mean_weight)
    else
      return -1e12
    end

    weights .= weights ./ sum(weights)

    ess_t = 1.0 / sum(weights .^ 2)
    total_ess += ess_t
    if ess_t < min_ess
      min_ess = ess_t
    end

    indices = systematic_resample(weights, N_particles)
    x_resampled = x_prev[indices]
    x_prev_prev_resampled = x_prev_prev[indices]

    # -------------------------------------------------------------------
    # Full-posterior jitter step, incorporating the leverage effect
    # -------------------------------------------------------------------
    if ess_t < (N_particles / 2.0)
      emp_std = std(x_resampled)
      sigma_jitter = max(emp_std * 0.20, 0.05)

      Threads.@threads for i in 1:N_particles
        # target density of each particle:
        # p(x_{t-1} | x_{t-2}, y_{1:t}) ∝ p(x_{t-1} | x_{t-2}, y_{t-1}) p(y_t | x_{t-1})
        x_curr = x_resampled[i]
        x_prop = x_curr + randn() * sigma_jitter
        prev_state = x_prev_prev_resampled[i]

        if t > 1
          # Condition the transition on the previously observed y_{t-1}
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
          # Fallback for t=1 where y_{t-1} does not exist
          drift_base = params.κ * (params.θ - prev_state) * dt
          expected_mean = prev_state + drift_base
          transition_std = params.σ_v * sqrt(dt)
        end

        # Evaluates p(x_{t-1} | y_{t-1}, x_{t-2})
        log_trans_prop = log_2pi_half - log(transition_std) - 0.5 * ((x_prop - expected_mean) / transition_std)^2
        log_trans_curr = log_2pi_half - log(transition_std) - 0.5 * ((x_curr - expected_mean) / transition_std)^2

        # Observation density evaluation relies on y_t
        v_prop = exp(x_prop)
        cont_mean_prop = (params.μ - 0.5 * (v_prop / 100.0)) * dt
        cont_var_prop = v_prop * dt

        v_curr = exp(x_curr)
        cont_mean_curr = (params.μ - 0.5 * (v_curr / 100.0)) * dt
        cont_var_curr = v_curr * dt

        # Evaluates p(y_t | x_{t-1})
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
    # Propagates particles
    # -------------------------------------------------------------------
    Threads.@threads for i in 1:N_particles
      x_curr = x_resampled[i]
      v_eval = exp(x_curr)

      drift_penalty = 0.5 * (v_eval / 100.0)
      cont_mean = (params.μ - drift_penalty) * dt

      # Extract the realized observation error (eta)
      if !is_fomc
        obs_var = v_eval * dt
        eta = y_t - cont_mean
      else
        jump_mean = params.β_0 + params.β_1 * shock
        obs_var = v_eval * dt + params.σ_ε^2
        eta = y_t - cont_mean - jump_mean
      end

      # Calculate conditional moments of ΔW_V given the price shock
      cov_Wv_eta = sqrt(v_eval) * params.ρ * dt
      cond_mean_Wv = (cov_Wv_eta / obs_var) * eta
      cond_var_Wv = dt - (cov_Wv_eta^2 / obs_var)

      # Draw correlated noise
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

    # Store variables for the next step's jitter look-back
    y_prev = y_t
    shock_prev = shock
    dt_prev = dt
    is_fomc_prev = is_fomc
  end

  if diagnostic
    avg_ess = total_ess / T
    println("\n[Particle Filter Diagnostics]")
    println("Target Particles (N): $N_particles")
    println("Average ESS:        $(round(avg_ess, digits=2))")
    println("Minimum ESS hit:    $(round(min_ess, digits=2))")
    println("Swarm Survival Rate: $(round((avg_ess/N_particles)*100, digits=2))%")
  end

  return log_likelihood
end

end # module MacroFinanceModel
