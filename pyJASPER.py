import numpy as np
from numpy.linalg import inv
from scipy.stats import invgamma
from scipy.special import gammaln, expit
from polyagamma import random_polyagamma


class JASPER:
    """
    The JASPER MCMC sampler class.

    Model:

        C_ij | Y_ij, phi_j ~ NB(mu_ij, phi_j)
        mu_ij = N_i exp(Y_ij)

        Y_ij = alpha_j + X_i^T beta_j + lambda_j^T f_i + eps_ij
        eps_ij ~ N(0, psi_j)

        f_i ~ N(0, I_r)

        lambda_j ~ N(0, tau_lambda^2 I_r)

        beta_j | gamma_j = 0 = 0
        beta_j | gamma_j = 1, g ~ N(0, g (X'X)^(-1))

        gamma_j | delta ~ Bernoulli(delta)
        delta ~ Beta(c_gamma, d_gamma)

        psi_j ~ IG(a_psi, b_psi)

        phi_j ~ Gamma(a_phi, b_phi)

        g ~ IG(a_g, b_g)
    """

    def __init__(
        self,
        C,
        X,
        N,
        r=5,
        n_iter=2000,
        burn=1000,
        thin=1,
        sigma_alpha2=100.0,
        tau_lambda2=1.0,
        a_psi=2.0,
        b_psi=1.0,
        a_phi=2.0,
        b_phi=0.1,
        phi_proposal_sd=0.10,
        a_g=0.05,
        b_g=0.05,
        c_gamma=1.0,
        d_gamma=1.0,
        seed=123,
    ):
        np.random.seed(seed)

        # -----------------------------
        # Data
        # -----------------------------
        self.C = np.asarray(C, dtype=float)      # n x p count matrix
        self.X = np.asarray(X, dtype=float)      # n x K spatial basis matrix
        self.N = np.asarray(N, dtype=float)      # n-vector library sizes

        self.n, self.p = self.C.shape
        self.K = self.X.shape[1]
        self.r = r

        # -----------------------------
        # MCMC settings
        # -----------------------------
        self.n_iter = n_iter
        self.burn = burn
        self.thin = thin
        self.n_keep = (n_iter - burn) // thin

        # -----------------------------
        # Hyperparameters
        # -----------------------------
        self.sigma_alpha2 = sigma_alpha2
        self.tau_lambda2 = tau_lambda2

        self.a_psi = a_psi
        self.b_psi = b_psi

        self.a_phi = a_phi
        self.b_phi = b_phi
        self.phi_proposal_sd = phi_proposal_sd

        self.a_g = a_g
        self.b_g = b_g

        self.c_gamma = c_gamma
        self.d_gamma = d_gamma

        # -----------------------------
        # Precompute fixed X quantities
        # -----------------------------
        self.G = self.X.T @ self.X
        self.G_inv = inv(self.G)

        # -----------------------------
        # Initialize parameters
        # -----------------------------
        self.Y = np.log((self.C + 0.5) / self.N[:, None])

        self.alpha = self.Y.mean(axis=0)

        self.B = np.zeros((self.K, self.p))

        self.gamma = np.ones(self.p, dtype=int)

        self.g = 1.0

        # Factor scores F: n x r
        self.F = np.random.normal(0.0, 1.0, size=(self.n, self.r))

        # Factor loadings Lambda: p x r
        self.Lambda = np.random.normal(
            0.0,
            np.sqrt(self.tau_lambda2),
            size=(self.p, self.r),
        )

        # Residual variances psi_j
        self.psi = np.ones(self.p)

        # NB dispersions phi_j
        self.phi = np.ones(self.p) * 10.0

        # -----------------------------
        # Storage
        # -----------------------------
        self.samples_gamma = np.zeros((self.n_keep, self.p), dtype=int)
        self.samples_phi = np.zeros((self.n_keep, self.p))
        self.samples_psi = np.zeros((self.n_keep, self.p))
        self.samples_g = np.zeros(self.n_keep)
        self.samples_alpha = np.zeros((self.n_keep, self.p))

        self.phi_accept_count = np.zeros(self.p)

    # =========================================================
    # Utility: log full conditional for phi_j
    # =========================================================

    def log_phi_conditional(self, phi, C_j, Y_j):
        """
        Log full conditional for phi_j up to constants.

        NB parameterization:
            mean = N_i exp(Y_ij)
            variance = mean + mean^2 / phi_j
        """

        if phi <= 0:
            return -np.inf

        mu = self.N * np.exp(Y_j)

        val = np.sum(
            gammaln(C_j + phi)
            - gammaln(phi)
            + phi * np.log(phi)
            - (C_j + phi) * np.log(phi + mu)
        )

        # Gamma(a_phi, b_phi), shape-rate
        val += (self.a_phi - 1.0) * np.log(phi) - self.b_phi * phi

        return val

    # =========================================================
    # Step 1: Polya-Gamma variables and pseudo-data
    # =========================================================

    def sample_pg_and_z(self):
        """
        For NB likelihood, define

            eta_ij = Y_ij + log N_i - log phi_j.

        Then

            omega_ij | - ~ PG(C_ij + phi_j, eta_ij).

        Conditional on omega_ij,

            z_ij | Y_ij, omega_ij ~ N(Y_ij, omega_ij^{-1}).
        """

        eta = self.Y + np.log(self.N[:, None]) - np.log(self.phi[None, :])

        b_pg = self.C + self.phi[None, :]

        omega = random_polyagamma(b_pg, eta)

        kappa = 0.5 * (self.C - self.phi[None, :])

        z = (
            kappa / omega
            - np.log(self.N[:, None])
            + np.log(self.phi[None, :])
        )

        return omega, z

    # =========================================================
    # Step 2: Update Y_ij
    # =========================================================

    def update_Y(self, omega, z):
        """
        Scalar Gaussian update for every Y_ij.

        Prior:
            Y_ij | - ~ N(m_ij, psi_j)

        PG pseudo-likelihood:
            z_ij | Y_ij ~ N(Y_ij, omega_ij^{-1})

        Therefore:
            Y_ij | - ~ N(mean_ij, var_ij)
        """

        spatial_mean = self.X @ self.B                 # n x p
        factor_mean = self.F @ self.Lambda.T           # n x p

        M = (
            self.alpha[None, :]
            + spatial_mean
            + factor_mean
        )

        precision = omega + 1.0 / self.psi[None, :]

        var = 1.0 / precision

        mean = var * (
            omega * z
            + M / self.psi[None, :]
        )

        self.Y = mean + np.sqrt(var) * np.random.normal(size=(self.n, self.p))

    # =========================================================
    # Step 3: Update factor scores F
    # =========================================================

    def update_F(self):
        """
        For each spot i:

            f_i | - ~ N(m_f, V_f)

        Since psi and Lambda are common across spots, V_f is the same
        for all i in a given MCMC iteration.
        """

        psi_inv = 1.0 / self.psi

        # V_f = (I_r + Lambda' Psi^{-1} Lambda)^(-1)
        Lambda_weighted = self.Lambda * psi_inv[:, None]

        V_f = inv(
            np.eye(self.r)
            + self.Lambda.T @ Lambda_weighted
        )

        # Residual after removing intercept and spatial component
        R = self.Y - self.alpha[None, :] - self.X @ self.B

        # For all i:
        # m_fi = V_f Lambda' Psi^{-1} r_i
        M_f = (R * psi_inv[None, :]) @ self.Lambda @ V_f.T

        L = np.linalg.cholesky(V_f)

        self.F = M_f + np.random.normal(size=(self.n, self.r)) @ L.T

    # =========================================================
    # Step 4: Update factor loadings Lambda
    # =========================================================

    def update_Lambda(self):
        """
        For each gene j:

            lambda_j | - ~ N(m_lambda_j, V_lambda_j)

        where residual removes alpha_j and X beta_j.
        """

        FtF = self.F.T @ self.F

        for j in range(self.p):
            r_j = self.Y[:, j] - self.alpha[j] - self.X @ self.B[:, j]

            precision = FtF / self.psi[j] + np.eye(self.r) / self.tau_lambda2

            V = inv(precision)

            mean = V @ (self.F.T @ r_j) / self.psi[j]

            self.Lambda[j, :] = np.random.multivariate_normal(mean, V)

    # =========================================================
    # Step 5: Update alpha_j
    # =========================================================

    def update_alpha(self):
        """
        For each gene j:

            alpha_j | - ~ N(m_alpha_j, V_alpha_j)
        """

        factor_part = self.F @ self.Lambda.T
        spatial_part = self.X @ self.B

        for j in range(self.p):
            r_j = self.Y[:, j] - spatial_part[:, j] - factor_part[:, j]

            precision = self.n / self.psi[j] + 1.0 / self.sigma_alpha2

            V = 1.0 / precision

            mean = V * np.sum(r_j) / self.psi[j]

            self.alpha[j] = np.random.normal(mean, np.sqrt(V))

    # =========================================================
    # Marginal likelihood pieces for gamma_j update
    # =========================================================

    def log_marginal_gamma0(self, r_j, psi_j):
        """
        r_j | gamma_j = 0 ~ N(0, psi_j I_n)
        """

        return (
            -0.5 * self.n * np.log(2.0 * np.pi)
            -0.5 * self.n * np.log(psi_j)
            -0.5 * np.dot(r_j, r_j) / psi_j
        )

    def log_marginal_gamma1(self, r_j, psi_j):
        """
        r_j | gamma_j = 1 after integrating beta_j:

            r_j ~ N(0, psi_j I + g X (X'X)^(-1) X')

        Using g-prior identities:

            log |psi I + g P_X|
              = n log psi + K log(1 + g / psi)

            r' V^{-1} r
              = r'r / psi
                - g / (psi * (psi + g)) r' P_X r
        """

        Xtr = self.X.T @ r_j

        proj_quad = Xtr.T @ self.G_inv @ Xtr

        rtr = np.dot(r_j, r_j)

        logdet = (
            self.n * np.log(psi_j)
            + self.K * np.log(1.0 + self.g / psi_j)
        )

        quad = (
            rtr / psi_j
            - (self.g / (psi_j * (psi_j + self.g))) * proj_quad
        )

        return (
            -0.5 * self.n * np.log(2.0 * np.pi)
            -0.5 * logdet
            -0.5 * quad
        )

    # =========================================================
    # Step 6: Update gamma_j and beta_j
    # =========================================================

    def sample_beta_active(self, r_j, psi_j):
        """
        beta_j | gamma_j = 1, -.

        Prior:
            beta_j ~ N(0, g G^{-1})

        Likelihood:
            r_j ~ N(X beta_j, psi_j I)

        Since G = X'X is fixed:

            V_beta = (psi^{-1} G + g^{-1} G)^(-1)
                   = psi*g/(psi+g) G^{-1}

            m_beta = g/(psi+g) G^{-1} X' r_j
        """

        V = (psi_j * self.g / (psi_j + self.g)) * self.G_inv

        mean = (
            self.g / (psi_j + self.g)
        ) * self.G_inv @ (self.X.T @ r_j)

        return np.random.multivariate_normal(mean, V)

    def update_gamma_and_B(self):
        """
        Uses the requested inactive-to-active odds:

            A_j =
              [p(Y | gamma_j=0, -) / p(Y | gamma_j=1, -)]
              *
              [(d + p - 1 - q_-j) / (c + q_-j)]

            pi_j = 1 / (1 + A_j)

            gamma_j | - ~ Bernoulli(pi_j)
        """

        factor_part = self.F @ self.Lambda.T

        for j in range(self.p):
            q_minus = int(self.gamma.sum() - self.gamma[j])

            # Residual after removing non-spatial effects
            r_j = self.Y[:, j] - self.alpha[j] - factor_part[:, j]

            log_p0 = self.log_marginal_gamma0(r_j, self.psi[j])
            log_p1 = self.log_marginal_gamma1(r_j, self.psi[j])

            log_prior_odds_0_to_1 = (
                np.log(self.d_gamma + self.p - 1 - q_minus)
                - np.log(self.c_gamma + q_minus)
            )

            log_A = (log_p0 - log_p1) + log_prior_odds_0_to_1

            pi_j = expit(-log_A)

            self.gamma[j] = np.random.binomial(1, pi_j)

            if self.gamma[j] == 0:
                self.B[:, j] = 0.0
            else:
                self.B[:, j] = self.sample_beta_active(r_j, self.psi[j])

    # =========================================================
    # Step 7: Update psi_j
    # =========================================================

    def update_psi(self):
        """
        Residual variance update:

            psi_j | - ~ IG(a_psi + n/2,
                           b_psi + 0.5 sum_i residual_ij^2)
        """

        M = (
            self.alpha[None, :]
            + self.X @ self.B
            + self.F @ self.Lambda.T
        )

        E = self.Y - M

        shape = self.a_psi + 0.5 * self.n

        for j in range(self.p):
            scale = self.b_psi + 0.5 * np.dot(E[:, j], E[:, j])
            self.psi[j] = invgamma.rvs(a=shape, scale=scale)

    # =========================================================
    # Step 8: Update global slab scale g
    # =========================================================

    def update_g(self):
        """
        g | beta, gamma ~ IG(a_g + Kq/2,
                            b_g + 0.5 sum beta_j' G beta_j)
        """

        active = np.where(self.gamma == 1)[0]
        q = len(active)

        if q == 0:
            self.g = invgamma.rvs(a=self.a_g, scale=self.b_g)
            return

        quad = 0.0

        for j in active:
            quad += self.B[:, j].T @ self.G @ self.B[:, j]

        shape = self.a_g + 0.5 * self.K * q
        scale = self.b_g + 0.5 * quad

        self.g = invgamma.rvs(a=shape, scale=scale)

    # =========================================================
    # Step 9: Update NB dispersions phi_j
    # =========================================================

    def update_phi(self):
        """
        Log-scale random-walk MH:

            log phi_j* = log phi_j + epsilon
            epsilon ~ N(0, proposal_sd^2)

        Acceptance includes Jacobian phi*/phi.
        """

        for j in range(self.p):
            phi_old = self.phi[j]

            log_phi_star = np.log(phi_old) + np.random.normal(
                0.0,
                self.phi_proposal_sd,
            )

            phi_star = np.exp(log_phi_star)

            log_old = self.log_phi_conditional(
                phi_old,
                self.C[:, j],
                self.Y[:, j],
            )

            log_new = self.log_phi_conditional(
                phi_star,
                self.C[:, j],
                self.Y[:, j],
            )

            log_acc = log_new - log_old + np.log(phi_star) - np.log(phi_old)

            if np.log(np.random.rand()) < log_acc:
                self.phi[j] = phi_star
                self.phi_accept_count[j] += 1

    # =========================================================
    # Main MCMC loop
    # =========================================================

    def run(self, verbose=True, show_every=50):
        keep_idx = 0

        for it in range(self.n_iter):
            # 1. Polya-Gamma augmentation
            omega, z = self.sample_pg_and_z()

            # 2. Latent Gaussian expression
            self.update_Y(omega, z)

            # 3. Low-rank factor scores
            self.update_F()

            # 4. Factor loadings
            self.update_Lambda()

            # 5. Intercepts
            self.update_alpha()

            # 6. SVG indicators and spatial coefficients
            self.update_gamma_and_B()

            # 7. Residual variances
            self.update_psi()

            # 8. Slab scale
            self.update_g()

            # 9. NB dispersions
            self.update_phi()

            # Store posterior samples
            if it >= self.burn and ((it - self.burn) % self.thin == 0):
                self.samples_gamma[keep_idx, :] = self.gamma
                self.samples_phi[keep_idx, :] = self.phi
                self.samples_psi[keep_idx, :] = self.psi
                self.samples_g[keep_idx] = self.g
                self.samples_alpha[keep_idx, :] = self.alpha
                keep_idx += 1

            if verbose and (it + 1) % show_every == 0:
                print(
                    f"iter {it + 1:5d}/{self.n_iter} | "
                    f"active={self.gamma.sum():3d} | "
                    f"mean(phi)={self.phi.mean():.3f} | "
                    f"mean(psi)={self.psi.mean():.3f} | "
                    f"g={self.g:.3f}"
                )

        self.phi_accept_rate = self.phi_accept_count / self.n_iter

        return self

    # =========================================================
    # Posterior summaries
    # =========================================================

    def posterior_inclusion_probabilities(self):
        return self.samples_gamma.mean(axis=0)

    def posterior_phi_mean(self):
        return self.samples_phi.mean(axis=0)

    def posterior_psi_mean(self):
        return self.samples_psi.mean(axis=0)

    def posterior_g_mean(self):
        return self.samples_g.mean()

    def selected_genes_by_threshold(self, threshold=0.5):
        ppi = self.posterior_inclusion_probabilities()
        return np.where(ppi >= threshold)[0]

    def selected_genes_by_pefdr(self, target=0.05):
        ppi = self.posterior_inclusion_probabilities()

        thresholds = np.sort(np.unique(ppi))[::-1]

        chosen = None

        for t in thresholds:
            selected = ppi >= t

            if selected.sum() == 0:
                continue

            pefdr = np.sum(1.0 - ppi[selected]) / selected.sum()

            if pefdr <= target:
                chosen = t

        if chosen is None:
            chosen = 1.0

        selected = np.where(ppi >= chosen)[0]

        return selected, chosen