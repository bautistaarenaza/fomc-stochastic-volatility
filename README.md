# FOMC Monetary Policy Surprises and Intraday Stochastic Volatility

Bayesian estimation of a stochastic volatility model with jumps, applied to intraday
equity returns, to measure how FOMC monetary policy surprises affect the level and the
dynamics of volatility.

The model is estimated by **particle marginal Metropolis–Hastings** (PMMH): an adaptive
random-walk Metropolis sampler in the parameter space, where the intractable likelihood
is replaced at each step by the unbiased estimate produced by a particle filter. The
filter is fully adapted — particles are weighted by the observation density *before*
propagation, and the leverage effect is handled by conditioning the volatility Brownian
increment on the realized return innovation.

## The model

The price $S_t$ and its latent variance $V_t$ evolve continuously between announcements. At
the instant of an FOMC release ($t = \tau$) a counting process $N_t$ fires, inducing a
discontinuity in both. Working with the log-price $y_t = \log S_t$ and the log-variance
$x_t = \log V_t$ — the latter guaranteeing $V_t > 0$ without constraints on $x_t$ — the
system is

$$dy_t = \left(\mu - \tfrac{1}{2}e^{x_t}\right)dt + e^{x_t/2} dW_t^S + Z_t^S dN_t$$

$$dx_t = \kappa(\theta - x_t) dt + \sigma_v dW_t^V + Z_t^V dN_t$$

where $-\tfrac{1}{2}e^{x_t}$ is the Itô correction, which keeps the expected rate of return on
$S_t$ equal to $\mu$ regardless of the volatility level. The two Wiener processes are
correlated,

$$\mathbb{E}\left[dW_t^S dW_t^V\right] = \rho dt,$$

with $\rho < 0$ capturing the leverage effect: sharp price falls tend to coincide with rises
in volatility.

Rather than treating the jump sizes as latent random variables, both are tied to an
observable monetary surprise $\Delta i_\tau$:

$$Z_\tau^S = \beta_0 + \beta_1 \Delta i_\tau + \epsilon_\tau, \qquad \epsilon_\tau \sim \mathcal{N}(0, \sigma_\epsilon^2)$$

$$Z_\tau^V = \alpha_0 + \alpha_1 |\Delta i_\tau|$$

The intercepts $\beta_0$ and $\alpha_0$ are the **pure-event** component: the impact of the
release itself, present even when the decision is fully anticipated ($\Delta i_\tau = 0$). The
slopes $\beta_1$ and $\alpha_1$ are the **magnitude-sensitive** component. The volatility jump
depends on $|\Delta i_\tau|$, so surprises in either direction raise volatility. No noise term
enters $Z_\tau^V$: since the variance is itself latent, an additional shock there would be
statistically redundant with the increment of $W_t^V$.

### Discretization

Data arrive at intervals of $\Delta t$, so an Euler–Maruyama scheme turns the system into
difference equations. With $\Delta y_t = y_t - y_{t-1}$ the observed log return over the
$t$-th window and $I_t \in \{0,1\}$ indicating an FOMC announcement in that window:

$$\Delta y_t = \left(\mu - \tfrac{1}{2}e^{x_{t-1}}\right)\Delta t + e^{x_{t-1}/2}\sqrt{\Delta t} \varepsilon_t^S + Z_t^S I_t$$

$$x_t = x_{t-1} + \kappa(\theta - x_{t-1})\Delta t + \sigma_v\sqrt{\Delta t} \varepsilon_t^V + Z_t^V I_t$$

where $\varepsilon_t = (\varepsilon_t^S, \varepsilon_t^V)^\top \sim \mathcal{N}(0, \Sigma)$ with

$$\Sigma = \begin{pmatrix} 1 & \rho \\\\ \rho & 1 \end{pmatrix}.$$

Because $\Delta y_t$ is observed, the particle filter recovers the realized $\varepsilon_t^S$
and draws the volatility innovation from its conditional law
$\varepsilon_t^V \mid \varepsilon_t^S \sim \mathcal{N}(\rho \varepsilon_t^S, 1 - \rho^2)$,
so simulated trajectories reproduce the leverage effect rather than ignoring it.

Ten parameters in total:
$\Theta = \{\mu, \kappa, \theta, \sigma_v, \rho, \beta_0, \beta_1, \sigma_\epsilon, \alpha_0, \alpha_1\}$.

> **Units note.** Returns are scaled to percentage points ($\Delta y_t = 100 \times$ log
> return) and $\Delta t$ is expressed in years, so $V_t$ carries units of
> $(\mathrm{p.p.})^2/\mathrm{yr}$. The Itô correction therefore appears in the code as
> `0.5 * (V / 100.0)`: under the rescaling $y = 100 r$, the term $100 \cdot \tfrac{1}{2}V_{\text{raw}}$
> becomes $\tfrac{1}{2}V/100$.

## Data

**Intraday equity prices** are drawn from the [Alpaca Market Data API](https://alpaca.markets/).
The pipeline downloads adjusted 10-minute bars and aggregates them into custom 30-minute
windows anchored at 09:50 ET, discarding overnight returns and any window spanning a data
gap. Returns are then normalized by a time-of-day volatility profile to remove the
intraday U-shape. Default tickers are SPY, IWM, and SHY.

**Monetary policy surprises** come from
[`marekjarocinski/jkshocks_update_fed`](https://github.com/marekjarocinski/jkshocks_update_fed),
Jarociński and Karadi's updated Fed surprise series. This project uses
`source_data/fomc_surprises_jk.csv`, which contains narrow-window (30-minute) surprises
around FOMC announcements since 1988. The relevant column is **MP1**, the current-month
Fed Funds futures surprise constructed from FF1 and FF2 following Gürkaynak, Sack and
Swanson (2005), measured in percentage points per annum. Only scheduled rate decisions are
retained. Each announcement is matched to the 30-minute bar containing it.

## Repository structure

```
fomc-stochastic-volatility/
├── code/
│ ├── MacroFinanceModel.jl # Module: particle filter with leverage and jumps
│ ├── gen_data_30m.jl # Build the 30-minute dataset
│ ├── run_pmcmc.jl # PMMH sampler
│ ├── rank_norm_gelman_rubin.jl # Convergence diagnostics
│ ├── compute_pit_vals.jl # Probability integral transforms
│ ├── test_pit_vals.jl # Uniformity and independence tests
│ ├── plot_marginal_densities.jl # Posterior marginals and summary tables
│ └── plot_stylized_jump.jl # FOMC-window returns vs. predictive band
├── data/
│ ├── fomc_surprises_jk.csv # JK surprise series (input)
│ └── SPY/ IWM/ SHY/ # Raw and processed price data (generated)
├── mcmc_chains/ # Posterior draws (generated)
├── pit_vals/ # PIT sequences (generated)
├── figs/ # Figures (generated)
├── Project.toml
└── Manifest.toml
```

## Setup

Install the pinned environment from the repository root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

The data download requires Alpaca credentials in the environment:

```bash
export ALPACA_KEY="your_key"
export ALPACA_SECRET="your_secret"
```

**All scripts are run from inside `code/`**, since data paths are relative to that
directory:

```bash
cd code
julia --project=@. gen_data_30m.jl
```

The `@.` form walks up the directory tree to find `Project.toml`, so the root environment
is picked up from the subdirectory.

## Pipeline

Each stage writes to disk, so the steps can be run independently once their inputs exist.
Set the `symbol` variable at the top of each script to choose the ticker.

### 1. Build the dataset — `gen_data_30m.jl`

Downloads 10-minute bars from Alpaca (cached to `data/<SYMBOL>/<SYMBOL>_10m_raw.csv` so
re-runs skip the API), filters to adjusted regular trading hours with early-close handling,
aggregates into 30-minute windows, computes log returns, tags FOMC bars with their MP1
surprise, and applies the time-of-day normalization.

*Output:* `data/<SYMBOL>/<SYMBOL>_30m.csv`

### 2. Estimate the model — `run_pmcmc.jl` + `MacroFinanceModel.jl`

Runs the PMMH sampler. `MacroFinanceModel.jl` provides the particle filter that returns
the unbiased log-likelihood estimate; `run_pmcmc.jl` wraps it in an adaptive-covariance
Metropolis–Hastings loop with a data-driven prior, an overdispersed starting point, and
periodic checkpointing so long runs can be resumed after an interruption.

Run this several times per ticker to obtain multiple independent chains — the convergence
diagnostics in step 3 need at least two.

*Output:* `mcmc_chains/<SYMBOL>/chain_N<particles>_L<length>_<k>.csv`

### 3. Check convergence — `rank_norm_gelman_rubin.jl`

Reports the rank-normalized, folded Gelman–Rubin statistic (Vehtari et al., 2021) for
every parameter: a global $\hat{R}$ across chains and a split $\hat{R}$ within each chain.
Values below 1.01 indicate convergence.

*Output:* console table

### 4. Evaluate the fit — `compute_pit_vals.jl`, then `test_pit_vals.jl`

`compute_pit_vals.jl` computes the probability integral transform
$u_t = F(y_t \mid y_{1:t-1}, \Theta)$ at each observation, averaging over thinned posterior
draws so that parameter uncertainty is integrated out rather than conditioned away.

`test_pit_vals.jl` then tests the two implications of correct specification: a
Kolmogorov–Smirnov test for uniformity of the $u_t$, and Ljung–Box tests on the
inverse-normal-transformed residuals and their squares for independence in the mean and
variance equations (Diebold, Gunther and Tay, 1998).

*Output:* `pit_vals/<SYMBOL>_pit_sequence.csv`; console tests; ACF and empirical-CDF
figures in `figs/`

### 5. Plot posterior marginals — `plot_marginal_densities.jl`

Pools all chains per ticker, derives interpretable quantities from the raw parameters
(mean-reversion half-life in trading days, the stationary mean and standard deviation of
$\sqrt{V_t}$, FOMC jumps as percentage changes in volatility), and plots the marginal
densities overlaid across tickers alongside printed posterior summary tables.

*Output:* `figs/posterior_continuous_all.pdf`, `figs/posterior_jumps_all.pdf`

### 6. Plot the FOMC jump — `plot_stylized_jump.jl`

Scatters realized returns in FOMC announcement windows against the corresponding monetary
surprises, overlaid with the posterior median of $\beta_0 + \beta_1 \Delta i_\tau$ and a 95%
posterior predictive band.

*Output:* `figs/stylized_jump_plot.pdf`

## Notes

- The particle filter is multi-threaded. Start Julia with `julia -t auto` (or set
 `JULIA_NUM_THREADS`) or the sampler will run single-threaded and be considerably slower.
- Estimation is the expensive step: 9,000 PMMH iterations at 2¹⁴ particles over a decade of
 30-minute bars takes hours. Checkpoints are written every 250 iterations and removed on
 successful completion.
- Generated data, chains, and PIT files are excluded from version control; everything under
 `data/<SYMBOL>/`, `mcmc_chains/`, and `pit_vals/` is reproducible from the scripts.

## References

Diebold, F. X., Gunther, T. A. and Tay, A. S. (1998). Evaluating density forecasts with
applications to financial risk management. *International Economic Review*, 39(4), 863–883.

Gürkaynak, R. S., Sack, B. and Swanson, E. (2005). Do actions speak louder than words? The
response of asset prices to monetary policy actions and statements. *International Journal
of Central Banking*, 1(1), 55–93.

Jarociński, M. and Karadi, P. (2020). Deconstructing monetary policy surprises — the role of
information shocks. *American Economic Journal: Macroeconomics*, 12(2), 1–43.
[doi:10.1257/mac.20180090](https://doi.org/10.1257/mac.20180090)

Vehtari, A., Gelman, A., Simpson, D., Carpenter, B. and Bürkner, P.-C. (2021).
Rank-normalization, folding, and localization: an improved $\hat{R}$ for assessing
convergence of MCMC. *Bayesian Analysis*, 16(2), 667–718.

## Data attribution

The FOMC surprise series is redistributed from
[`marekjarocinski/jkshocks_update_fed`](https://github.com/marekjarocinski/jkshocks_update_fed)
under CC BY 4.0. If you use it, cite Jarociński and Karadi (2020). Intraday price data is
retrieved from Alpaca under their terms of service and is not redistributed here.
