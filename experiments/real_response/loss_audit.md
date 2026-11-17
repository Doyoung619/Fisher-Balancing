# Exact GLM objective and numerical audit

The v2 learner minimizes `sum_i ell(y_i, x_i @ theta) + rho/2 * ||theta||^2`, with fixed `rho=1` and every coordinate, including the intercept, penalized. A mean-loss implementation would require ridge `rho/n`; this implementation directly uses the sum objective. The full-pool reference uses the same fixed ridge. Each matched reference uses `rho*n_pool/n_total` and is evaluation-only.

| Family | Loss without likelihood constants | Mean | Exact curvature |
|---|---|---|---|
| Bernoulli | `logaddexp(0,eta)-y*eta` | `expit(eta)` | `expit(eta)*expit(-eta)` |
| Gaussian working model | `(y-eta)^2/2` | `eta` | `1` |
| Poisson working model | `exp(eta)-y*eta` | `exp(eta)` | `exp(eta)` |

For binary labels, the Bernoulli implementation uses the algebraically identical `logaddexp(0, -eta)` when y=1 and `logaddexp(0, eta)` when y=0. This avoids subtraction cancellation. Gaussian full NLL adds `log(2*pi)/2`; Poisson full NLL adds `gammaln(y+1)`. Gaussian squared-residual loss differs from the canonical `eta^2/2-y*eta` only by the recorded-response constant `y^2/2`. Variance is fixed at one.

The gradient is `X.T @ (mu-y) + rho*theta`. The Hessian is `(X.T * w) @ X + rho*I`. Gaussian fits use a Cholesky linear solve. Bernoulli and Poisson fits use damped Newton with Cholesky directions and Armijo backtracking, warm-started from the previous checkpoint. Nonfinite trial objectives are rejected. Same-objective BFGS followed by Newton polishing is the retry path, with original attempts retained in `fit_attempts`. No predictor, response, loss or exact likelihood curvature is clipped.

Learner convergence requires `||gradient||_inf / (1 + ||X.T@y||_inf + rho*||theta||_inf) <= 1e-8`; reference convergence requires `<=1e-9`. Small steps and iteration-limit exits cannot independently mark convergence. The numerator, denominator, relative residual, Newton decrement, iterations, line-search trials, retries, objective decomposition, curvature extrema and Hessian condition are retained. The Armijo comparison permits an objective-roundoff allowance of `8*eps*(1+abs(objective))`, while the final stationarity gate is unchanged. Single-class Bernoulli fits remain well-defined with positive ridge and do not introduce fake observations.

Acquisition alone may floor Bernoulli/Poisson curvature at `1e-12`. Primary fits, V_ref and V_hat always use exact unfloored curvature. V_acq records the numerical acquisition modification, including floor count, fraction, added mass and operator-norm difference. At every checkpoint V_hat reweights **all** selected rows at current theta. Gaussian V_ref, V_hat, V_acq and H_design coincide. Positive ridge ensures nonzero information denominators. Data-information rank tolerance is `max(1, largest regularized eigenvalue)*1e-10`; negative data-information eigenvalues outside that numerical allowance are flagged and never silently clipped.

Poisson evaluation computes deviance using `y*log(y)-y*eta-y+exp(eta)`, with the y=0 logarithmic term set by `xlogy(0,0)=0`. True prediction overflow remains INF and is flagged; it is not replaced with a finite clipped answer. Full-pool fixed/matched and selected-data objective gaps retain signed roundoff and are checked for materially negative values.

Validation command: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -m pytest -q tests/test_numerics.py tests/test_aggregate_risk.py`.

Measured validation on 2026-10-04: **14 tests passed**. The tests check analytic loss/gradient/Hessian by central finite differences for all families, sum/mean ridge scaling, Gaussian normal equations, strict learner/reference stationarity, single-class logistic fits, Poisson overflow without clipping, exact information/Hessian equality, Gaussian matrix equality, Weyl tracking inequality, nonnegative same-objective optimizer gaps, matched/fixed equality at full pool, and additive reference information. Risk tests separately check uncentered Q, denominator S, bias decomposition, mean-L2 versus RMS versus directional risk, rank limits, degenerate top-eigenspace projectors, leave-one-seed-out values and paired whole-seed resampling. Tests use explicitly synthetic numerical fixtures; experimental responses remain recorded data.

Source mapping: manuscript Eq. (4) is the objective/gradient audit; Eq. (9) motivates the two working-information matrices; Eq. (15) motivates uncentered finite-seed Q/R2 squared/Rdir squared; Eq. (35) motivates normalized tracking. Replacing unknown truth with a fixed empirical full-pool reference and using finite paired algorithmic seeds are new empirical design decisions. They do not verify true-parameter asymptotic constants or the full-ball theorem on misspecified real responses.
