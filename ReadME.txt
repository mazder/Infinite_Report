To run the program, from the top of the directory run the following commands on linux terminal

	python3 Comparison_Sampling_Methods.py
	
	
Sample Output:
python3 Comparison_Sampling_Methods.py 
/PATH/Comparison_Sampling_Methods.py:29: IntegrationWarning: The integral is probably divergent, or slowly convergent.
  z, error = quad(calculate_exp_term, -np.inf, np.inf, args=(mu, sigma, alpha, beta))
Rejection Sampling: Estimated E[x]: -0.71543, Runtime: 0.15599 seconds
Metropolis-Hastings Sampling: Estimated E[x]: 26.92299, Runtime: 0.02159 seconds
Gibbs Sampling: Estimated E[x]: -17.09821, Runtime: 0.18944 seconds
Inverse Transform Sampling: Estimated E[x]: 7.94424, Runtime: 0.01064 seconds
Standard Gaussian Sampling: Estimated E[x]: 0.92209, Runtime: 0.00003 seconds
{'Rejection Sampling': <function rejection_sampling at 0x771f236fde10>, 'Metropolis-Hastings Sampling': <function metropolis_hastings_sampling at 0x771f236fdea0>, 'Gibbs Sampling': <function gibbs_sampling at 0x771f236fe050>, 'Inverse Transform Sampling': <function inverse_transform_sampling at 0x771f236fe0e0>, 'Standard Gaussian Sampling': <function standard_gaussian_sampling at 0x771f236fe170>}
[-0.7154283852471239, 26.922992553374655, -17.098213676870458, 7.944239694018204, 0.9220887748924245]
[0.15599274635314941, 0.02158951759338379, 0.189439058303833, 0.010640144348144531, 3.147125244140625e-05]


