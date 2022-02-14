function SignalOut = Manakov_GPU(SignalIn,P)

% Two axis Adaptive Manakov Equation Solver (now fixed step)
% Symmeterised Split step NLSE solver, includes 2nd and 3rd order
% dispersion, attenuation and Kerr non-linearities.
%
% SignalOut=Manakov_Adaptive(SignalIn,P)
%
% Inputs:
% SignalIn - input signal structure
% P - Fibre parameters structure contains
%  .Length - fibre length (km)
%  .RefWavelength - reference wavelength (nm)
% Only specify the following parameters if the simulation requires it
%  .Att - fibre attenuation (dB/km)
%  .D - dispersion parameter at reference wavelength (ps/nm/km)
%  .S - dispersion slope at reference wavelength (ps/nm^2/km)
%  .Gamma - nonlinear parameter (/W/km)
%  .PhiMax - Maximum nonlinear phase shift per nonlinear step - (rads)
%
% Returns:
% SignalOut - output signal structure
%
% Author: D Millar - Aug. 2010

%File: Manakov_Adaptive.m
%Date: 18 April 2012
%Author: David Ives
%email: d.ives@ee.ucl.ac.uk

% this version replaces B Thomson version which was a modified version of this
% but did not work.
% 16/3/12 some corrections made to allow for zero dispersion and zero attenuation
% 17/4/12 power estimation updated to include both polarisations where available
% 18/4/12 speed optimised - now only for dual polarisation signals 60% time
% is spent on ffts.

%File: Manakov_GPU.m
%Date: 27 April 2012
%Author: David Ives
%email: d.ives@ee.ucl.ac.uk

% updated based on Sean Kilmurray NLSE_GPU for use on GPU
% returned to fixed step, input P.dz

SignalOut=SignalIn;
 
c=299792.458;              % nm/ps

[Np,Nt] = size(SignalIn.Et); 
dF = SignalIn.Fs/(Nt*10^12);                    % Spectral resolution (THz)
FF = [0:(Nt/2)-1,-Nt/2:-1] * dF;            % Frequency array (THz)

Nz=ceil(P.Length/P.dz); 
dz=P.Length/Nz; %km fixed steps

if isfield(P, 'Att')&&(P.Att~=0)
    a=gpuArray(P.Att*log(10)/10);         % /km
else
    a=gpuArray(0);
end

if isfield(P, 'D')
    B2=-P.D*P.RefWavelength.^2/(2*pi*c);   	         % ps^2/km
    d2=gpuArray(1i/2*B2*(2*pi*FF.').^2);
else
    d2=gpuArray(zeros(size(FF.')));
end

if isfield(P, 'S')
    B3=P.RefWavelength.^2/(2*pi*c).^2*(P.RefWavelength.^2*P.S+2*P.RefWavelength*P.D);      % ps^3/km
    d3=gpuArray(1i*B3/6*(2*pi*FF.').^3);
else
    d3=gpuArray(zeros(size(FF.')));
end

if a>0
  	dzEff = (1-exp(-a*dz))/a;
else
   	dzEff=dz;
end

D1 = exp(dz/2*(d2+d3));
D=[D1, D1];

Ef=gpuArray(ifft(SignalIn.Et.'));      % Fourier Transform to frequency domain
%disp('Using Two Axis Adaptive Split Step NLSE with Manakov Nonlinear estimation - GPU')
    
for i=1:Nz	
	Ef=Ef.*D;             % Apply Dispersion operator for the first half step
	Et=fft(Ef);                            % Fourier Transform to time domain
	power = sum(real(Et).^2+imag(Et).^2,2); % Total Intermediate power (W)
    N1=exp(dzEff*1i*P.Gamma*8/9*power-dz*a/2);    % Nonlinear operator 
	N=[N1, N1];
	Et=Et.*N;          % Apply Nonlinear and loss operators at center of step
	Ef=ifft(Et);                      % Fourier Transform to frequency domain
    Ef=Ef.*D;            % Apply Dispersion operator for the second half step
end

SignalOut.Et=gather(fft(Ef)).';   % Fourier Transform to time domain
