# Continuous-ball reference and theory audit

The supplied manuscript is the source for the sum-loss constrained ridge MLE (eq:mle), Fisher weights (eq:fisher), full-history recomputed plug-in information (eq:states), uncentered error second moment (eq:risks), and fixed-positive-coefficient temperature analysis. Its archived finite-pool settings are not used. The proof of prop:efficiency uses uniqueness of the *matrix*, not uniqueness of its representing design measure.

## Conditions and qualifications

Here the parameter set is the radius-4 closed ball and all specified truths have norm 0.75, 1.5, or 3, strictly interior. Unit-variance Gaussian curvature is one; Bernoulli curvature is positive on the compact predictor range [-4L,4L] and Lipschitz there. Centered Gaussian errors have sub-Gaussian scale 1; centered Bernoulli errors have scale 1/2. Repeated queries receive fresh independent responses, revealed after the entire batch. With exact continuous acquisition and the stated estimator, these meet the manuscript's full-ball regularity conditions. Numerical acquisition and estimator errors remain numerical qualifications: tests cannot promote a locally checked optimizer to the exact maximization assumed by the theorem.

We set the new protocol's N_cal=400; it is a design choice, not a value forced by the manuscript. tau0 is fixed thereafter. The manuscript main eq:temperature has no floor, while its appendix still mentions a 1e-8 floor. This textual inconsistency has no numerical effect here: every actual tau_range is greater than 1e-8. We use the main equation without any floor, clipping, or coefficient reset. Radius changes hold kappa fixed, change the truth norm, and retain rho=1; finite-N exact scale invariance is not claimed.

## Gaussian full-ball reference

For every probability measure, tr M <= L² and lambda_min M <= L²/d. The equally weighted radius-L orthonormal vectors attain M=(L²/d)I. Any maximizing M must have all eigenvalues at least L²/d and their sum at most L², so M is unique. Hence gamma=L²/d and tr(M^-1)=d²/L².

## Bernoulli rotational reduction and boundary completion

Write theta*=kappa u/L with unit u. Average an arbitrary feasible measure under every orthogonal transformation fixing u. Concavity and rotational invariance of lambda_min show that this does not lower the criterion. The resulting information has form a uu^T+b(I-uu^T). At fixed longitudinal coordinate z=u^T x, logistic curvature is fixed. Replacing the transverse radius by sqrt(L²-z²), with a uniform transverse direction, preserves a and does not decrease b. Thus it suffices to consider distributions of z on [-L,L] with a(z)=w(kappa z/L)z² and b(z)=w(kappa z/L)(L²-z²)/(d-1). This argument is for computing a reference; acquisition is never rotated using the truth.

## Analytic global dual certificate for every requested condition

Let t=z/L, t0=1/sqrt(d), c=kappa*tanh(kappa/(2sqrt(d)))/(2sqrt(d)), q=[1+(d-1)c]/d, and A=(1-q)/(d-1)=(1-c)/d. In every requested condition 0<c<1, so P=q uu^T+A(I-uu^T) is strictly positive definite with trace one. The dual score divided by L² is

f(t)=w(kappa t)[A+c t²].

It is even. For t>0 its derivative has the sign of c h(t)-A, where

h(t)=2t/[kappa*tanh(kappa t/2)]-t².

Writing a=kappa t/2 gives h'(t)<0 because coth(a)-a*csch(a)²<2a, which is exactly tanh(2a)<2a after multiplication by sinh(a)². The chosen c makes the derivative zero at t0; consequently this is the unique maximizer on [0,1], including endpoints by continuity. The global dual bound is therefore gamma=L²*w(kappa/sqrt(d))/d.

A feasible measure with z=L/sqrt(d), uniform transverse covariance, has a=b=gamma. The implementation constructs it explicitly using the 2(d-1) support points L*u/sqrt(d) +/- L*sqrt(1-1/d)*v_j, with v_j an orthonormal basis of u-perp and equal weights. Thus the lower and upper bounds coincide analytically. A separate adaptive scalar support LP starts from 129 equally spaced longitudinal coordinates, refines the support through continuous scalar searches using its own LP dual, and verifies a feasible lower bound within relative 1e-7. That numerical scalar search supplies refinement, not the global upper certificate. No adaptive trajectory is used in either computation.

`src/reference.py` also evaluates the analytic formula with 50-decimal interval arithmetic and outward binary64 endpoint conversion; those gamma_lower/gamma_upper values bound numerical representation error, independently of the LP lower check. Saved reconstructed support information, dual minimum eigenvalue, and residual norms audit the actual floating point matrices.

## Uniqueness among all optimal matrices

The tight dual satisfies tr(P M)<=gamma for every feasible matrix, including nonsymmetric matrices before rotational averaging. Any optimal M has M-gamma I positive semidefinite. Since P is strictly positive definite and tr P=1, optimality gives tr[P(M-gamma I)]=0, forcing M-gamma I=0. Hence uniqueness is analytically established for every requested Bernoulli matrix as well as Gaussian. The isotropic formula tr(M^-1)=d/gamma is justified by this argument, not assumed from rotational symmetry alone.

For an approximately tight numerical certificate with residual epsilon and lambda_min(P)=p_min>0, the excess positive-semidefinite mass obeys tr(M-gamma I)<=epsilon/p_min. The saved dual eigenvalues and primal residuals make that numerical margin auditable.

## Metrics and interpretation

At N=mT, Q_hat=mean_r(e_r e_r^T) is uncentered, with no dimension division. Information eigenvalues are computed per replicate before averaging. The directional eigenvalue is computed only after averaging outer products. The four panels are, in order: true lambda_min/N, plug-in lambda_min/N, N*gamma*lambda_max(Q_hat), N*tr(Q_hat). True information has finite-N ceiling gamma+rho/N; plug-in information has no corresponding ceiling. The two sharp-risk horizontal lines apply to FB under the manuscript's conditions and exact-oracle idealization; they are not universal baseline limits or a universal A-optimal lower bound. Ridge-biased finite-N risk may be below them.

SEM is reported for information means and mean squared Euclidean error from independent repetition IDs. Directional stability uses disjoint blocks/prefixes and recomputes the entire largest-eigenvalue statistic under resampling. Descriptive bootstrap ranges are not guaranteed 95% confidence intervals at tied eigenvalues. There is no centering, shrinkage, eigenvalue debiasing, selective repetition removal, or trajectory fitting to set a reference. The final report separates finite-N discrepancy, finite-repetition uncertainty, independent reference error, and acquisition/estimation numerical error.
