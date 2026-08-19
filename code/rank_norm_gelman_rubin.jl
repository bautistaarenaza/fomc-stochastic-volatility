# ==========================================
# rank_norm_gelman_rubin.jl
# ==========================================
using CSV
using DataFrames
using Statistics
using StatsBase      # tiedrank: tie-averaged ranks for rank normalization
using Distributions  # inverse normal CDF (quantile of Normal) for the Blom transform
using Printf

# ==========================================
# --- MAIN CONFIGURATION ---
# ==========================================
symbol = "SPY"

# ==========================================
# 1. FIND AND LOAD ALL CHAIN FILES
# ==========================================
println("Scanning directory for chain files...")

# Find all files starting with "chain_" and ending with ".csv"
chains_dir = "../mcmc_chains/" * symbol * "/"

if !isdir(chains_dir)
  error("Directory $chains_dir does not exist. Run run_pmcmc.jl first.")
end

all_files = readdir(chains_dir)
chain_files = filter(f -> startswith(f, "chain_") && endswith(f, ".csv"), all_files)

if isempty(chain_files)
  error("No chain files found in $chains_dir. Run run_pmcmc.jl first.")
end

println("Found $(length(chain_files)) chain files:")
chains = Vector{DataFrame}()

for (i, file) in enumerate(chain_files)
  println(" -> Loading: $file")
  df = CSV.read(chains_dir * file, DataFrame)
  push!(chains, df)
end

num_chains = length(chains)
param_names = names(chains[1])

# Report log_posterior first, ahead of the individual model parameters
if "log_posterior" in param_names
  param_names = filter(x -> x != "log_posterior", param_names)
  pushfirst!(param_names, "log_posterior")
end

num_params = length(param_names)
chain_length = nrow(chains[1])

# The global diagnostic stacks the chains into one matrix, so they must all be
# the same length and a single chain leaves the between-chain variance undefined
if any(nrow(df) != chain_length for df in chains)
  error("Chain files have differing lengths: $(nrow.(chains)). The global R̂ requires equal-length chains.")
end

if num_chains < 2
  error("Only one chain found. The global R̂ requires at least two independent chains.")
end

println("\nLoaded $num_chains chains of $chain_length draws across $num_params quantities.")

# ==========================================
# 2. RANK-NORMALIZED GELMAN-RUBIN R-HAT
# ==========================================

# Classical R-hat math. This is used only as an inner building block:
# it is applied to *rank-normalized* draws rather than to the raw draws.
function rhat_core(psi::AbstractMatrix)
  n, m = size(psi)
  chain_means = mean(psi, dims=1)
  overall_mean = mean(chain_means)

  # Between-chain variance (B)
  B = (n / (m - 1)) * sum((chain_means .- overall_mean) .^ 2)

  # Within-chain variance (W)
  chain_vars = var(psi, dims=1)
  W = mean(chain_vars)

  # Estimated marginal variance (V_hat)
  V_hat = ((n - 1) / n) * W + (1 / n) * B

  # R-hat statistic
  return sqrt(V_hat / W)
end

# Rank normalization (Vehtari et al. 2021).
# Pool every draw, replace each one by its tie-averaged rank within the pool,
# then map those ranks to normal scores via the Blom inverse-normal transform.
# The original matrix shape is preserved so the output feeds directly into
# rhat_core. This makes R-hat invariant to monotone transforms and robust to
# heavy-tailed / non-normal marginals.
function rank_normalize(psi::AbstractMatrix)
  S = length(psi)
  ranks = tiedrank(vec(psi))            # ranks across ALL chains pooled together
  c = 3 / 8                             # Blom offset
  p = (ranks .- c) ./ (S - 2c + 1)      # plotting positions strictly in (0, 1)
  z = Distributions.quantile.(Normal(), p)
  return reshape(z, size(psi))
end

# Rank-normalized R-hat reported as max(bulk, folded-tail), matching the
# `posterior` R package / ArviZ recommendation.
#  - bulk: rank-normalize the draws, then classical R-hat (sensitive to the location).
#  - tail: fold around the pooled median (|x - median|), rank-normalize, recompute
#          (sensitive to disagreement in spread / tails).
function rhat_rank_normalized(psi::AbstractMatrix)
  rhat_bulk = rhat_core(rank_normalize(psi))

  folded = abs.(psi .- median(psi))
  rhat_tail = rhat_core(rank_normalize(folded))

  return max(rhat_bulk, rhat_tail)
end

# Global rank-normalized R-hat across all chains
function calculate_global_rhat(chains_data::Vector{DataFrame}, param::String)
  m = length(chains_data)
  n = nrow(chains_data[1])

  # Extract the chains for this specific parameter into a matrix (n x m)
  psi = zeros(n, m)
  for j in 1:m
    psi[:, j] = chains_data[j][!, param]
  end

  return rhat_rank_normalized(psi)
end

# Split rank-normalized R-hat for a single chain. Comparing a chain's two
# halves against each other detects drift that a between-chain comparison
# alone would miss.
function calculate_split_rhat(df::DataFrame, param::String)
  vals = df[!, param]
  n_total = length(vals)
  half = div(n_total, 2)

  # Create a matrix with 2 columns: first half and second half
  psi = hcat(vals[1:half], vals[(half+1):(2*half)])

  return rhat_rank_normalized(psi)
end

# ==========================================
# 3. REPORT
# ==========================================
println("\n--- RANK-NORMALIZED GELMAN-RUBIN DIAGNOSTICS (R̂) ---")
println("(Rank-normalized & folded; values < 1.01 indicate convergence)")

# Build the header dynamically, keeping the interpolated pieces out of the
# loop body to avoid soft-scope issues at top level
header_base = @sprintf("%-15s | %-8s", "Parameter", "Global R̂")
header_splits = join([@sprintf(" | Split R̂ (C%d)", i) for i in 1:num_chains], "")
full_header = header_base * header_splits

println(full_header)
println("-"^length(full_header))

# Calculate and print metrics per parameter
for param in param_names
  global_rhat = calculate_global_rhat(chains, param)

  row_base = @sprintf("%-15s | %.4f  ", param, global_rhat)
  row_splits = join([@sprintf(" | %.4f       ", calculate_split_rhat(chains[c], param)) for c in 1:num_chains], "")

  println(row_base * row_splits)
end
println("-"^length(full_header), "\n")
