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

    m_ = eqAnisoMeanMagneticMomentAlongZAxis(B_, DCore, MS, temp, kAnis_, order, epsilon)
   
    # the field is rotated into the original coordinate system
    for ti=1:length(tVec)  
      y[ti, :] = R * vec(m_[ti,:])
    end
  else
    dt = step(tVec)/10000

    B_ = zeros(Float64, length(tVec), 3, 2)
    for ti=1:length(tVec)  
      B_[ti,:,1] = R' * vec(B(tVec[ti]))
      B_[ti,:,2] = R' * vec(B(tVec[ti]+dt))
    end

    m1_ = eqAnisoMeanMagneticMomentAlongZAxis(B_[:,:,1], DCore, MS, temp, kAnis_, order, epsilon)
    m2_ = eqAnisoMeanMagneticMomentAlongZAxis(B_[:,:,2], DCore, MS, temp, kAnis_, order, epsilon)

    # the field is rotated into the original coordinate system
    for ti=1:length(tVec)  
      y[ti, :] = R * (vec(m2_[ti,:]) .- vec(m1_[ti,:])) ./ dt
    end
  end
                  
  return y
end


gammaln(A) = log(gamma(A))

function eqAnisoMeanMagneticMomentAlongZAxis(H, DCore, MS, temp, kAnis::Real, order, epsilon)
  return eqAnisoMeanMagneticMomentAlongZAxis(H, DCore, MS, temp, repeat(kAnis:kAnis,size(H,1)), order, epsilon)
end

function eqAnisoMeanMagneticMomentAlongZAxis(H, DCore, MS, temp, kAnis::AbstractVector, order, epsilon)
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
  is_alpha_k_larger_zero = alphak .>= epsilon
  is_alpha_k_zero = (!).(is_alpha_k_larger_zero)
  
  betaH = beta*H;
  
  # Check if alpha_K is large, otherwise use series for \alpha_K towards zero.
  #  \xi in Laguerre polynomials L^{(\alpha))(\xi) with  \xi=-\beta^2\tilde{H}^2_3/4/\alpha_K
  squared_beta_H3_alphak = (-(betaH[is_alpha_k_larger_zero,3]).^2/4.0./alphak[is_alpha_k_larger_zero]);
  # alpha_K is very small
  #  Limit Laguerre polynomials: \limit \xi^+ to 0 for
  #  xi^l*L^{(\alpha))_l(-\beta^2 \tilde{H}^2_3/\xi)  is
  #  (\beta^2 \tilde{H}^2_3)^l/4^l/\Gamma(l+1)
  squared_beta_H3 = (betaH[is_alpha_k_zero,3]).^2/4;
  log_squared_beta_H3 = log.(squared_beta_H3);

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
  laguerrePoly12 = ones(length(squared_beta_H3_alphak),order+2);
  #log_laguerrePoly32 = ones(length(beta_normH_12),order+2);
  log_laguerrePoly12 = ones(length(log_squared_beta_H3),order+2);
  
  # for each series term the calculation is done in logarithmic scale.
  # the calculation uses a recursive representation of the Laguerre polynomials 
  # and modified Bessel functions are evaluated directly 
  # The series are calculated from k=0 to order+1 
  # Note that k==l in IWMPI2022 paper
  #@showprogress  
  for k = 0:order+1
      besselFuncTerm[:,k+1] = log.(besseli.(1/2+k,beta_normH_12)) # I_{1/2+l}(\beta|H|_{12}) in log-scale
      #v[:,k+1] = log(besseli(1/2+k,beta_normH_12,1))+beta_normH_12; %use of rescaled besseli 
                                                                      #with multiplication of exp(-beta_normH_12) possible to reduce overflow 
                                                                      #if beta_normH_12 is large (not used)
      besselFuncTerm[is_H_12_nearly_zero,k+1] .= 0 # if \beta|H|_{12} is approximately zero use limits
      factor = (-1/2+k)*log_2+log_pi_sqrt # 2^{-1/2+k}*sqrt(pi) in log-scale
  
      # check if \alpha_K is approximately zero
      #if alphak<epsilon
          # no anisotropy present -> should correspond to the equilibrium model 
          #                          without anisotropy (Langevin function)
          if k==0
            #initizalize
            polyLimit  = zeros(size(log_squared_beta_H3))
          else
            polyLimit = k*log_squared_beta_H3  # (\beta H_3)^{2k} in log-scale
          end
          
          # Limits of Laguerre polynomials for \alpha_k to 0
          value = polyLimit .- gammaln(k+1);
          log_laguerrePoly12[:,k+1] = value # (beta H_3)^{2k}/4^k/\Gamma(l+1)
          #log_laguerrePoly32(:,k+1) = value;% (beta H_3)^{2k}/4^k/\Gamma(l+1)
  
          powerTerms32[is_alpha_k_zero,k+1] = factor .- (3/2+k).*log_beta_normH_12[is_alpha_k_zero] #  2^{-1/2+k}*sqrt(pi)/|H|_{12}^{3/2+k}
          powerTerms32[is_H_12_nearly_zero .& is_alpha_k_zero,k+1] .= factor+BesselLimit32[k+1] # sqrt(pi)*2^{-1/2+k}*2^{-3/2-k}/\Gamma(5/2+k)
          powerTerms12[is_alpha_k_zero,k+1] = factor .- (1/2+k).*log_beta_normH_12[is_alpha_k_zero] # 2^{-1/2+k}*sqrt(pi)/|H|_{12}^{1/2+k}
          powerTerms12[is_H_12_nearly_zero .& is_alpha_k_zero,k+1] .= factor+BesselLimit12[k+1] #  2^{-1/2+k}*sqrt(pi)*2^{-1/2-k}/\Gamma(3/2+k)= 2^{-1}*sqrt(pi)/(\Gamma(3/2+k))
     # else
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
          temp_ = k*log_alphak .+ factor # \alpha_K^k*2^{-1/2+k}*sqrt(pi)
          powerTerms32[is_alpha_k_larger_zero,k+1] .= temp_[is_alpha_k_larger_zero] .-(3/2+k)*log_beta_normH_12[is_alpha_k_larger_zero]  # \alpha_K^k*2^{-1/2+k}*sqrt(pi)/(|H|_{12})^{3/2+k}
          powerTerms32[is_H_12_nearly_zero .& is_alpha_k_larger_zero,k+1] .= 
              temp_[is_H_12_nearly_zero .& is_alpha_k_larger_zero].+BesselLimit32[k+1]  # \alpha_K^k*2^{-1/2+k}*sqrt(pi)2^{-3/2-k}/\Gamma(5/2+k) =\alpha_K^k 2^{-2}*sqrt(pi)/\Gamma(5/2+k)
          powerTerms12[is_alpha_k_larger_zero,k+1] .= temp_[is_alpha_k_larger_zero] .-(1/2+k)*log_beta_normH_12[is_alpha_k_larger_zero]   # \alpha_K^k*2^{-1/2+k}*sqrt(pi)/(|H|_{12})^{1/2+k}
          powerTerms12[is_H_12_nearly_zero .& is_alpha_k_larger_zero,k+1] .= 
              temp_[is_H_12_nearly_zero .& is_alpha_k_larger_zero] .+ BesselLimit12[k+1] # \alpha_K^k*2^{-1/2+k}*sqrt(pi)*2^{-1/2-k}/\Gamma(3/2+k) = sqrt(pi)/2*\alpha_K^k/\Gamma(3/2+k)
    #  end
  end
  
  betaH3 = betaH[:,3]
  log_laguerrePolyMinus12 = zeros(length(beta_normH_12), order+2);
  z3 = zeros(length(beta_normH_12),1);
  #if alphak>=epsilon
  # Calculated the other Laguerre polynomials of order \alpha=-1/2  from
  # order \alpha = 1/2 Laguerre polynomials
  # Note : L^{(-1/2)}_{n+1}(x)=L^{(1/2)}_{n+1}(x)- L^{(1/2)}_{n}(x)
  log_laguerrePolyMinus12[is_alpha_k_larger_zero,2:end] = log.(laguerrePoly12[:,2:end]-laguerrePoly12[:,1:end-1])
  # calculate the "unnormalized" magnetization in z-direction
  z3[is_alpha_k_larger_zero] .= betaH3[is_alpha_k_larger_zero].*sum(exp.(besselFuncTerm[is_alpha_k_larger_zero,2:end]+powerTerms32[is_alpha_k_larger_zero,1:end-1]+log.(laguerrePoly12[:,1:end-1])),dims=2)
  #else
  # calculate the magnetization in z-direction if approximately zero is nearly zero
  z3[is_alpha_k_zero] .= betaH3[is_alpha_k_zero].*sum(exp.(besselFuncTerm[is_alpha_k_zero,2:end]+powerTerms32[is_alpha_k_zero,1:end-1]+log_laguerrePoly12[:,1:end-1]),dims=2)
  
  # In Limit \alpha_K->0 limit the order of \alpha plays no role
  log_laguerrePolyMinus12[is_alpha_k_zero,:] = log_laguerrePoly12
  #end

  exponent_partition_function = besselFuncTerm+powerTerms12+log_laguerrePolyMinus12
  exponent_x_y = besselFuncTerm[:,2:end]+powerTerms12[:,1:end-1]+log_laguerrePolyMinus12[:,1:end-1]
  # calculated the partition function
  Z = sum(exp.(exponent_partition_function),dims=2)
  
  # calculated the "unnormalized"  magnetization in x- and y-direction
  resI = sum(exp.(exponent_x_y),dims=2)
  z2 = betaH2.*resI
  z1 = betaH1.*resI
  
  
  # H is nearly zero and \alpha_k is nearly zero
  #if alphak<epsilon
  # H = 0
  beta_H3_is_small = zeros(Bool,length(is_alpha_k_zero))
  beta_H3_is_small[is_alpha_k_zero] .= squared_beta_H3 .< eps2
  ind2 = is_alpha_k_zero .& is_H_12_nearly_zero .& beta_H3_is_small;
  #ind2 = is_H_12_nearly_zero .& (squared_beta_H3.<eps2)
  Z[ind2] .= 1
  z3[ind2] .= 0
  z2[ind2] .= 0
  z1[ind2] .= 0
  #end
  #end
  
  m = zeros(size(Z,1),3);
  # Normalize so that the mean magnetic moment is obtained
  m[:,1] = z1./Z
  m[:,2] = z2./Z
  m[:,3] = z3./Z
  
  return m # msat .* m
end



function eqAnisoMeanMagFullVec(H, DCore, MS, temp, kAnis, nEasy, order, epsilon)
  #EQANISOTROPMEANMAG Calculation of the mean magentic equilibrium moment for SPIOs
  #                   with an anistropy along a given easy axis
  #
  #  Input: H             magentic field (number of datapoints x 3)
  #         parameter     physical and particle parameter
  #         order         maximum order, where the infinte series should truncated
  #         epsilon       Value that defines when fields and K_aniso are defined as "small"  (default: epsilon=1e-10)
  #
  # Output: Mag           mean magnetic moment of H (number of datapoints x 3)
  
  nEasy_ = nEasy'
  # Splitting of the applied field H into contributions along the easy axis 
  # and the orthogonal contribution.
  H_easy_axis = sum(H.*(nEasy_),dims=2);
  H_orthogonal = H.-H_easy_axis.*nEasy_;
  
  # take the norm of the orthogonal field
  magnitude_H_orthogonal = sqrt.(vec(sum(H_orthogonal.^2,dims=2)));
  
  # z is the field contribution along the easy axis, only the magnitude is 
  # needed for the contribution of the orthogonal field
  tilde_H = hcat(zeros(size(magnitude_H_orthogonal)),magnitude_H_orthogonal,H_easy_axis);
  
  # the sign of the orthogonal field contribution of H
  sign_H_orthogonal = zeros(size(H_orthogonal));
  H_orthogonal_larger_zero = magnitude_H_orthogonal .> 0;
  sign_H_orthogonal[H_orthogonal_larger_zero,:] = H_orthogonal[H_orthogonal_larger_zero,:]./magnitude_H_orthogonal[H_orthogonal_larger_zero,:];
  
  # Calculate the mean magnetic moment
  Mag = eqAnisoMeanMagneticMomentAlongZAxis(tilde_H, DCore, MS, temp, kAnis, order, epsilon);

  # Addition of the two contributions to the mean magentic moment
  Mag = sign_H_orthogonal.*Mag[:,2] .+ nEasy_.*Mag[:,3];

  return Mag
end