

data { 
  int<lower=1> cond; // Number of conditions
  int<lower=1> S; // Number of subjects
  int<lower=1> npar; // Number of parameters
  int<lower=1> nlag; // Number of lags
  int<lower=0> nHits[cond,S,nlag];
  int<lower=0> nFA[cond,S,nlag];
  int<lower=0> nTargets[cond,S,nlag];
  int<lower=0> nDistractors[cond,S,nlag];
  real<lower=0> LAG[cond,S,nlag];
  matrix[cond, npar] x;
  int<lower=1> k; // runner for loglikelihood
}

parameters {
	vector[npar] a0;	
	vector[npar] b0;	
	vector[npar] c0;	
	real criterion0;

	matrix[npar*3+1,S] deltahat_tilde; //3 = A,B,C,1 = criterion

	cholesky_factor_corr[npar*3+1] L_Omega; 
	vector<lower=0>[npar*3+1] sigma; 

}

transformed parameters {

	real<lower=0,upper=1> HITrate[cond,S,nlag];
	real<lower=0,upper=1> FArate[cond,S,nlag];
	
	matrix[npar*3+1,S] deltahat; 
	matrix[npar,S] A;			// Intercept
	matrix[npar,S] B;			// Rate
	matrix[npar,S] C;			// Asymptote 
	
	real d[cond,S,nlag]; 			// dprime
	vector[cond] tmp_a;
	vector[cond] tmp_b;
	vector[cond] tmp_c;

	vector[S] criterion;
	deltahat = diag_pre_multiply(sigma, L_Omega) * deltahat_tilde; 	

	for (i in 1:S) {
		A[,i] = a0 + deltahat[1:npar,i];
		B[,i] = b0 + deltahat[(npar+1):(npar*2),i];
		C[,i] = c0 + deltahat[(2*npar+1):(npar*3),i];
		criterion[i] = criterion0 + deltahat[npar*3+1,i]; 
		
		tmp_a = x * A[,i];
		tmp_b = x * B[,i];
		tmp_c = x * C[,i];
		
		
		for (j2 in 1:cond) {
			for (j in 1:nlag) {
				if (LAG[j2,i,j] > tmp_a[j2]){
					d[j2,i,j] = (tmp_c[j2] * (1-exp ( -(LAG[j2,i,j] - tmp_a[j2]) * tmp_b[j2])));
				} else {
					d[j2,i,j]=0;
				}
				HITrate[j2,i,j] = Phi(d[j2,i,j] / 2 -criterion[i]);
				FArate[j2,i,j] = Phi(-d[j2,i,j] / 2 -criterion[i]);
			}			
		}		
	}

  }

model {

	for (i in 1:S) {
		for (j2 in 1:cond) {
			for (j in 1:nlag) {
				nHits[j2,i,j] ~ binomial(nTargets[j2,i,j], HITrate[j2,i,j]);
				nFA[j2,i,j] ~ binomial(nDistractors[j2,i,j], FArate[j2,i,j]);

			}
		}
	}

	L_Omega ~ lkj_corr_cholesky(1); 
	sigma ~ cauchy(0, 4); 
	to_vector(deltahat_tilde) ~ normal(0, 1); 
	
	a0[1] ~ normal(0.3,0.1) ;
	b0[1] ~ normal(2,1);
	c0[1] ~ normal(4,1);
	
	a0[2:] ~ normal(0,1) ;
	b0[2:] ~ normal(0,1);
	c0[2:] ~ normal(0,1);
	
	criterion0 ~ normal(0,1);
}

generated quantities {
  corr_matrix[npar*3+1] Omega;  
  int r;

real log_lik[cond*S*nlag*2];

r = k;

  Omega = L_Omega * L_Omega';

 	for (i in 1:S) {
		for (j2 in 1:cond) {
			for (j in 1:nlag) {
				log_lik[r] = binomial_log(nHits[j2,i,j],nTargets[j2,i,j],HITrate[j2,i,j]);
				r = r+1				;
				log_lik[r] = binomial_log(nFA[j2,i,j],nDistractors[j2,i,j],FArate[j2,i,j]);
				r = r+1	;
					  }
				   }
			}
}

