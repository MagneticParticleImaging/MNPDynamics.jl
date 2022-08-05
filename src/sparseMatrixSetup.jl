
function generateSparseMatricesNeel(N, p1, p2, p3, p4, τNeel)

  nz = calcnz(N)

  V = zeros(ComplexF64,nz);
  I = zeros(Int64,nz);
  J = zeros(Int64,nz);
  counter = 0;
  ind = 1;
  for r=0:N
    for q=-r:r
        counter = counter+1;
        I[ind] = counter;
        J[ind] = counter;
        V[ind] = -1/(2*τNeel)*r*(r+1) + p4*(r^2+r-3*q^2)/((2*r+3)*(2*r-1));
        ind = ind+1;
        if q!=-r
            if r!=0 && q!=r
                I[ind] = counter-2*r;
                J[ind] = counter;
                V[ind] = -1im * p3 * q*(r-q)/(2*r-1);
                ind = ind+1;
            end
        end
        if r<(N-1)
            I[ind] = counter+2*(r+1)+2*(r+2);
            J[ind] = counter;
            V[ind] = -p4*r*(r+q+1)*(r+q+2)/((2*r+3)*(2*r+5));
            ind = ind+1;
        end
        if r>1 && q>(-r+1)
            I[ind] = counter-2*r-2*(r-1);
            J[ind] = counter;
            V[ind] = p4*(r+1)*(r-q)*(r-q-1)/((2*r-3)*(2*r-1));
            ind = ind+1;
        end
        if r<N
            I[ind] = counter+2*(r+1);
            J[ind] = counter;
            V[ind] = -1im * p3 * q*(r+q+1)/(2*r+3);
            ind = ind+1;
        end
    end
  end

  K = findfirst(isequal(0.0), I) - 1
  m_offset = sparse(J[1:K], I[1:K], V[1:K], (N+1)^2, (N+1)^2);
  dropzeros!(m_offset)

  V = zeros(ComplexF64,nz);
  I = zeros(Int64,nz);
  J = zeros(Int64,nz);
  counter = 0;
  ind = 1;
  for r=0:N
    for q=-r:r
        counter = counter+1;
        I[ind] = counter;
        J[ind] = counter;
        V[ind] = -1im/2.0 * p1 *2*q;
        ind = ind+1;
        if q!=-r
            if r!=0 && q!=r
                I[ind] = counter-2*r;
                J[ind] = counter;
                V[ind] = p2*(r+1)*(r-q)/(2*r-1);
                ind = ind+1;
            end
        end
        if r<N
            I[ind] = counter+2*(r+1);
            J[ind] = counter;
            V[ind] = -p2*r*(r+q+1)/(2*r+3);
            ind = ind+1;
        end
    end
  end

  K = findfirst(isequal(0.0), I) - 1
  m_b3 = sparse(J[1:K], I[1:K], V[1:K], (N+1)^2, (N+1)^2);
  dropzeros!(m_b3)

  V = zeros(ComplexF64,nz);
  I = zeros(Int64,nz);
  J = zeros(Int64,nz);
  counter = 0;
  ind = 1;
  for r=0:N
    for q=-r:r
        counter = counter+1;
        if q!=r
            I[ind] = counter+1;
            J[ind] = counter;
            V[ind] = -1im/2.0 * p1 *(r-q)*(r+q+1);
            ind = ind+1;
        end
        if q<(r-1)
            I[ind] = counter-2*r+1;
            J[ind] = counter;
            V[ind] = p2 * (r+1)*(r-q)*(r-q-1)/(4*r-2);
            ind = ind+1;
        end
        if r<N
            I[ind] = counter+2*(r+1)+1;
            J[ind] = counter;
            V[ind] = p2 * r*(r+q+1)*(r+q+2)/(4*r+6);
            ind = ind+1;
        end
    end
  end

  K = findfirst(isequal(0.0), I) - 1
  m_bp = sparse(J[1:K], I[1:K], V[1:K], (N+1)^2, (N+1)^2);
  dropzeros!(m_bp)

  V = zeros(ComplexF64,nz);
  I = zeros(Int64,nz);
  J = zeros(Int64,nz);
  counter = 0;
  ind = 1;
  for r=0:N
    for q=-r:r
        counter = counter+1;
        if q!=-r
            I[ind] = counter-1;
            J[ind] = counter;
            V[ind] = -1im/2.0*p1;
            ind = ind+1;
        end
        if q>(-r+1)
            I[ind] = counter-2*r-1;
            J[ind] = counter;
            V[ind] = -p2 * (r+1)/(4*r-2);
            ind = ind+1;

        end
        if r<N
            I[ind] = counter+2*(r+1)-1;
            J[ind] = counter;
            V[ind] = -p2 * r/(4*r+6);
            ind = ind+1;
        end
    end
  end

  K = findfirst(isequal(0.0), I) - 1
  m_bm = sparse(J[1:K], I[1:K], V[1:K], (N+1)^2, (N+1)^2);
  dropzeros!(m_bm)
  
  return m_offset, m_b3, m_bp, m_bm
end




function generateSparseMatricesBrown(N, p2, τBrown)

  nz = calcnz(N)

  V = zeros(ComplexF64,nz);
  I = zeros(Int64,nz);
  J = zeros(Int64,nz);
  counter = 0;
  ind = 1;
  for r=0:N
    for q=-r:r
        counter = counter+1;
        I[ind] = counter;
        J[ind] = counter;
        V[ind] = -1/(2*τBrown)*r*(r+1)
        ind = ind+1;
    end
  end

  K = findfirst(isequal(0.0), I) - 1
  m_offset = sparse(J[1:K], I[1:K], V[1:K], (N+1)^2, (N+1)^2);
  dropzeros!(m_offset)

  V = zeros(ComplexF64,nz);
  I = zeros(Int64,nz);
  J = zeros(Int64,nz);
  counter = 0;
  ind = 1;
  for r=0:N
    for q=-r:r
        counter = counter+1;
        if q!=-r
            if r!=0 && q!=r
                I[ind] = counter-2*r;
                J[ind] = counter;
                V[ind] = p2*(r+1)*(r-q)/(2*r-1);
                ind = ind+1;
            end
        end
        if r<N
            I[ind] = counter+2*(r+1);
            J[ind] = counter;
            V[ind] = -p2*r*(r+q+1)/(2*r+3);
            ind = ind+1;
        end
    end
  end

  K = findfirst(isequal(0.0), I) - 1
  m_b3 = sparse(J[1:K], I[1:K], V[1:K], (N+1)^2, (N+1)^2);
  dropzeros!(m_b3)

  V = zeros(ComplexF64,nz);
  I = zeros(Int64,nz);
  J = zeros(Int64,nz);
  counter = 0;
  ind = 1;
  for r=0:N
    for q=-r:r
        counter = counter+1;
        if q<(r-1)
            I[ind] = counter-2*r+1;
            J[ind] = counter;
            V[ind] = p2 * (r+1)*(r-q)*(r-q-1)/(4*r-2);
            ind = ind+1;
        end
        if r<N
            I[ind] = counter+2*(r+1)+1;
            J[ind] = counter;
            V[ind] = p2 * r*(r+q+1)*(r+q+2)/(4*r+6);
            ind = ind+1;
        end
    end
  end

  K = findfirst(isequal(0.0), I) - 1
  m_bp = sparse(J[1:K], I[1:K], V[1:K], (N+1)^2, (N+1)^2);
  dropzeros!(m_bp)

  V = zeros(ComplexF64,nz);
  I = zeros(Int64,nz);
  J = zeros(Int64,nz);
  counter = 0;
  ind = 1;
  for r=0:N
    for q=-r:r
        counter = counter+1;
        if q>(-r+1)
            I[ind] = counter-2*r-1;
            J[ind] = counter;
            V[ind] = -p2 * (r+1)/(4*r-2);
            ind = ind+1;

        end
        if r<N
            I[ind] = counter+2*(r+1)-1;
            J[ind] = counter;
            V[ind] = -p2 * r/(4*r+6);
            ind = ind+1;
        end
    end
  end

  K = findfirst(isequal(0.0), I) - 1
  m_bm = sparse(J[1:K], I[1:K], V[1:K], (N+1)^2, (N+1)^2);
  dropzeros!(m_bm)
  
  return m_offset, m_b3, m_bp, m_bm
end






function calcnz(N)
  counter = 0;
  nz = 0;
  for r=0:N
    for q=-r:r
        counter = counter+1;

        nz = nz+1;
        if q!=-r
            nz = nz+1;
            if r!=0 && q!=r && q!=-r
                nz = nz+1;
            end
        end
        if q!=r
            nz = nz+1;
        end
        if r<(N-1)
            nz = nz+1;
        end
        if r>1 && q>(-r+1)
            nz = nz+1;
        end
        if q>(-r+1)
            nz = nz+1;
        end
        if q<(r-1)
            nz = nz+1;
        end
        if r<N
            nz = nz+1;
            nz = nz+1;
            nz = nz+1;
        end
    end
  end
  return nz
end