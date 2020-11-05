#ax1 predefined
dt = 0.1 # ms per bin
T = 2e3 # in ms



t = arange(0,T,dt)
spktr = zeros(t.shape)
k = 1/tauk*exp(-t/tauk)
spktr[[4000,4100,4200,4300,4400]] = 1
filtered_S = a*lfilter(k,1,spktr)+b
filtered_S2 = a2*lfilter(k,1,spktr)+b2
filtered_S = roll(filtered_S,1)
filtered_S2 = roll(filtered_S2,1)
if nlin=='sigmoid':
	mean_om = mufac*sigmoid(filtered_S)
	gam = gamfac*sigmoid(filtered_S2)
else:
	mean_om = mufac*(tanh(filtered_S)+1)/2.0
	gam = gamfac#*(tanh(filtered_S)+1)/2.0


I = zeros((Ntrial, len(spktr)))
for trial in range(Ntrial):
	wspktr = mean_om*spktr
	gspktr = gam*spktr
	for ind in where(wspktr>0)[0]:
		if model == 'gamma':
			wspktr[ind] = random.gamma(wspktr[ind]**2/gspktr[ind]**2,scale=gspktr[ind]**2/wspktr[ind])
	tau_psc = 5.0
	kappa=-1*exp(-t/tau_psc)
	I[trial,:] = lfilter(kappa,1,wspktr) + 0.05*random.randn(len(spktr))

