# The implementation of the equilibrium model with anisotropy was developed by
# Marco Maass and is based on this paper: https://dx.doi.org/10.18416/ijmpi.2022.2203008


function simulationMNP(B::g, tVec, ::EquilibriumAnisModel;
                       MS = 474000.0, 
                       DCore = 20e-9, 
                       temp = 293.0,
                       kAnis = 625,
                       derivative = false, 
                       order = 120, # order factor for the truncation of series expansion -> 
                                    #  to large values could rise NaNs or Inf
                                    # small order faster calculation, but lower probally lower accuracy
                       epsilon = 1e-10, # Value that defines when fields and kAnis are defined as "small"
                       kargs...
                       ) where g
    

  y = zeros(Float64, length(tVec), 3)

  if typeof(kAnis) <: AbstractVector || typeof(kAnis) <: Tuple
    kAnis_ = norm(collect(kAnis))
    n = norm(collect(kAnis)) > 0 ? normalize(collect(kAnis)) : [1.0, 0.0, 0.0]
  else
    kAnis_ = kAnis
    n = [0.0;0.0;1.0]
  end

  R = rotz(n)  
  if !derivative
    # The field is rotated so that the easy axis is along the z-axis

    B_ = zeros(Float64, length(tVec), 3)
    for ti=1:length(tVec)  
      B_[ti,:] = R' * vec(B(tVec[ti]))
    end

    m_ = eqAnisoMeanMagneticMomentAlongZAxis(B_, DCore, MS, temp, kAnis_, n, order, epsilon)
   
    # the field is rotated into the original coordinate system
    for ti=1:length(tVec)  
      y[ti, :] = R * vec(m_[ti,:])
    end

    st = step(tVec)

    kB = 1.38064852e-23
    gamGyro = 1.75*10.0^11
    VCore = π/6 * DCore^3

    tau = 3e-10* exp(kAnis_*VCore / (kB*temp)*(1-maximum(abs.(B_))/700e-3)^2)

    y = corrRelaxation(y, 1/st, tau)
  else
    error("Needs to be implemented")
  end
                  
  return y
end

function corrRelaxation(s, baseFreq, τ)
  N = size(s,1)
  S = rfft(s, 1)
  Q = size(S,1)
  freq = baseFreq*collect(0:(Q-1))./Q
  R = 1 ./ (1.0 .+ 1im*2*pi*freq*τ)
  S = S.*R
  scorr = irfft(S, size(s,1), 1)
  return scorr
end

gammaln(A) = log(gamma(A))

function eqAnisoMeanMagneticMomentAlongZAxis(H, DCore, MS, temp, kAnis, n, order, epsilon)
  #eqAnisoMeanMagneticMomentAlongZAxis    Calculation of the mean magentic equilibrium moment for SPIOs
  #                                       where the easy axis anisotropy is aligned along the z-axis.
  #
  #  Input: H             magentic field (number of datapoints x 3)
  #         parameter     physical and particle parameter
  #         order         maximum order, where the infinte series should truncated
  #         epsilon       Value that defines when fields and kAnis are defined as "small"  (default: epsilon=1e-10)
  #
  # Output: Mag           mean magnetic moment of H (number of datapoints x 3)
  
  
  eps2 = epsilon/10;
  eps3 = epsilon*1e4;
  
  kB = 1.38064852e-23
  gamGyro = 1.75*10.0^11
  VCore = π/6 * DCore^3
  μ₀ = 4*π*1e-7 #vacuum permeability
  msat = MS * VCore #saturation magnetic moment of a single nanoparticle
  beta = msat /(kB*temp) #H measured in T/μ₀
  
  F_c = VCore/kB/temp;
  
  # calculation mostly in logarithmic scale to avoid overflow and underflow
  log_2 = log(2);
  log_pi_sqrt = log(pi)/2;
  
  # Limit values if x->0
  # Limit_{x->0} BesselI[1/2+n,x]/(x)^(1/2+n) =  2^{-1/2-n}/\Gamma(3/2+n)
  BesselLimit12 = -gammaln.(3/2 .+(0:order+1)).-log_2.*((0:order+1).+1/2)
  # Limit_{x->0} BesselI[3/2+n,x]/(x)^(3/2+n) = 2^{-3/2-n}/\Gamma(5/2+n)
  BesselLimit32 = -gammaln.(5/2 .+(0:order+1)).-log_2.*((0:order+1).+3/2)
  
  # strength anisotropy \alpha_K as defined in "An analytical equilibrium solution to
  # the Neel relaxation Fokker-Planck equation" (IWMPI2022)
  alphak = F_c*kAnis
  log_alphak = log.(alphak)
  
  betaH = beta*H;
  
  # Check if alpha_K is large, otherwise use series for \alpha_K towards zero.
  if alphak >= epsilon
      #  \xi in Laguerre polynomials L^{(\alpha))(\xi) with  \xi=-\beta^2\tilde{H}^2_3/4/\alpha_K
      squared_beta_H3_alphak = (-(betaH[:,3]).^2/4/alphak);
  else
      #  Limit Laguerre polynomials: \limit \xi^+ to 0 for
      #  xi^l*L^{(\alpha))_l(-\beta^2 \tilde{H}^2_3/\xi)  is
      #  (\beta^2 \tilde{H}^2_3)^l/4^l/\Gamma(l+1)
      squared_beta_H3 = (betaH[:,3]).^2/4;
      log_squared_beta_H3 = log.(squared_beta_H3);
  end
  beta_normH_12 = sqrt.(sum(betaH[:,1:2].^2,dims=2));
  log_beta_normH_12 = log.(beta_normH_12);
  
  # check if norm \beta|\tilde{H}_{12}| is not zero
  is_H_12_nearly_zero = vec(beta_normH_12).<eps3;
  
  betaH2 = (betaH[:,2]./beta_normH_12);
  betaH2[is_H_12_nearly_zero] .= 0;
  betaH1 = (betaH[:,1]./beta_normH_12);
  betaH1[is_H_12_nearly_zero] .= 0;
  
  besselFuncTerm = zeros(length(beta_normH_12),order+2);
  powerTerms32 = zeros(length(beta_normH_12),order+2);
  powerTerms12 = zeros(length(beta_normH_12),order+2);
  laguerrePoly12 = ones(length(beta_normH_12),order+2);
  #log_laguerrePoly32 = ones(length(beta_normH_12),order+2);
  log_laguerrePoly12 = ones(length(beta_normH_12),order+2);
  
  # for each series term the calculation is done in logarithmic scale.
  # the calculation uses a recursive representation of the Laguerre polynomials 
  # and modified Bessel functions are evaluated directly 
  # The series are calculated from k=0 to order+1 
  # Note that k==l in IWMPI2022 paper
  for k = 0:order+1
      besselFuncTerm[:,k+1] = log.(besseli.(1/2+k,beta_normH_12)) # I_{1/2+l}(\beta|H|_{12}) in log-scale
      #v[:,k+1] = log(besseli(1/2+k,beta_normH_12,1))+beta_normH_12; %use of rescaled besseli 
                                                                      #with mulitplication of exp(-beta_normH_12) possible to reduce overflow 
                                                                      #if beta_normH_12 is large (not used)
      besselFuncTerm[is_H_12_nearly_zero,k+1] .= 0 # if \beta|H|_{12} is approximately zero use limits
      factor = (-1/2+k)*log_2+log_pi_sqrt # 2^{-1/2+k}*sqrt(pi) in log-scale
  
      # check if \alpha_K is approximately zero
      if alphak<epsilon
          # no anisotropy present -> should correspond to the equilibrium model 
          #                          without anisotropy (Langevin function)
          if k==0
              #initizalize
              temp_  = zeros(size(log_squared_beta_H3))
          else
              temp_ = k*log_squared_beta_H3  # (\beta H_3)^{2k} in log-scale
          end
          
          # Limits of Laguerre polynomials for \alpha_k to 0
          value = temp_ .- gammaln(k+1);
          log_laguerrePoly12[:,k+1] = value # (beta H_3)^{2k}/4^k/\Gamma(l+1)
          #log_laguerrePoly32(:,k+1) = value;% (beta H_3)^{2k}/4^k/\Gamma(l+1)
  
          powerTerms32[:,k+1] = factor .- (3/2+k).*log_beta_normH_12 #  2^{-1/2+k}*sqrt(pi)/|H|_{12}^{3/2+k}
          powerTerms32[is_H_12_nearly_zero,k+1] .= factor+BesselLimit32[k+1] # sqrt(pi)*2^{-1/2+k}*2^{-3/2-k}/\Gamma(5/2+k)
          powerTerms12[:,k+1] = factor .- (1/2+k).*log_beta_normH_12 # 2^{-1/2+k}*sqrt(pi)/|H|_{12}^{1/2+k}
          powerTerms12[is_H_12_nearly_zero,k+1] .= factor+BesselLimit12[k+1] #  2^{-1/2+k}*sqrt(pi)*2^{-1/2-k}/\Gamma(3/2+k)= 2^{-1}*sqrt(pi)/(\Gamma(3/2+k))
      else
          #anisotropy path
          if k==0
              # the zeroth degree Laguerre polyonmial with order \alpha=(1/2)
              # L^{(1/2)}_0(x)=1| x=squared_beta_H3_alphak is 1
          elseif k==1
              temp2 = 3/2 .- squared_beta_H3_alphak # L^{(1/2)}_1(x)  x=squared_beta_H3_alphak
              laguerrePoly12[:,2] = temp2
          else
              # recursive formula of the Lagurre polynomials for k>=1
              # L^{(1/2)}_k(x). This part can have large values.
              # Maybe optimization is possible.
              temp2 = 3/2 .- squared_beta_H3_alphak # L^{(1/2)}_1(x)  x=squared_beta_H3_alphak

              laguerrePoly12[:,k+1] = ((2*(k-1) .+ temp2).*laguerrePoly12[:,k].-(k-1/2).*laguerrePoly12[:,k-1])/(k);
          end
          temp_ = k*log_alphak+factor # \alpha_K^k*2^{-1/2+k}*sqrt(pi)
          powerTerms32[:,k+1] .= temp_ .-(3/2+k)*log_beta_normH_12  # \alpha_K^k*2^{-1/2+k}*sqrt(pi)/(|H|_{12})^{3/2+k}
          powerTerms32[is_H_12_nearly_zero,k+1] .= temp_+BesselLimit32[k+1]  # \alpha_K^k*2^{-1/2+k}*sqrt(pi)2^{-3/2-k}/\Gamma(5/2+k) =\alpha_K^k 2^{-2}*sqrt(pi)/\Gamma(5/2+k)
          powerTerms12[:,k+1] .= temp_ .-(1/2+k)*log_beta_normH_12   # \alpha_K^k*2^{-1/2+k}*sqrt(pi)/(|H|_{12})^{1/2+k}
          powerTerms12[is_H_12_nearly_zero,k+1] .= temp_+BesselLimit12[k+1] # \alpha_K^k*2^{-1/2+k}*sqrt(pi)*2^{-1/2-k}/\Gamma(3/2+k) = sqrt(pi)/2*\alpha_K^k/\Gamma(3/2+k)
      end
  end
  
  betaH3 = betaH[:,3]
  if alphak>=epsilon
      log_laguerrePolyMinus12 = zeros(size(laguerrePoly12));
      # Calculated the other Laguerre polynomials of order \alpha=-1/2  from
      # order \alpha = 1/2 Laguerre polynomials
      # Note : L^{(-1/2)}_{n+1}(x)=L^{(1/2)}_{n+1}(x)- L^{(1/2)}_{n}(x)
      log_laguerrePolyMinus12[:,2:end] = log.(laguerrePoly12[:,2:end]-laguerrePoly12[:,1:end-1])
      # calculate the "unnormalized" magenetization in z-direction
      z3 = betaH3.*sum(exp.(besselFuncTerm[:,2:end]+powerTerms32[:,1:end-1]+log.(laguerrePoly12[:,1:end-1])),dims=2)
  else
      # calculate the magenetization in z-direction if approximately zero is nearly zero
      z3 = betaH3.*sum(exp.(besselFuncTerm[:,2:end]+powerTerms32[:,1:end-1]+log_laguerrePoly12[:,1:end-1]),dims=2)
  
      # In Limit \alpha_K->0 limit the order of \alpha plays no role
      log_laguerrePolyMinus12 = log_laguerrePoly12
  end
  exponent_partition_function = besselFuncTerm+powerTerms12+log_laguerrePolyMinus12
  exponent_x_y = besselFuncTerm[:,2:end]+powerTerms12[:,1:end-1]+log_laguerrePolyMinus12[:,1:end-1]
  # calculated the partition function
  Z = sum(exp.(exponent_partition_function),dims=2)
  
  # calculated the "unnormalized"  magnetization in x- and y-direction
  resI = sum(exp.(exponent_x_y),dims=2)
  z2 = betaH2.*resI
  z1 = betaH1.*resI
  
  
  # H is nearly zero and \alpha_k is nearly zero
  if alphak<epsilon
      # H = 0
      ind2 = is_H_12_nearly_zero .& (squared_beta_H3.<eps2)
      Z[ind2] .= 1
      z3[ind2] .= 0
      z2[ind2] .= 0
      z1[ind2] .= 0
      #end
  end
  
  
  m = zeros(size(Z,1),3);
  # Normalize so that the mean magnetic moment is obtained
  m[:,1] = z1./Z
  m[:,2] = z2./Z
  m[:,3] = z3./Z
  
  return m # msat .* m
end