function Signal = Manakov(Signal, P)

% Two axis NLSE computed assuming fast-varying birefringence
% Symmeterised Split step NLSE solver, includes 2nd and 3rd order
% dispersion, PMD, SPM and XGM.
%
% Signal = Manakov(Signal, P)
%
% Inputs:
% SignalIn - input signal structure
% P - Fibre parameters structure contains
%  .Length - fibre length (km)
%  .dz - simulation step size (km)
%  .RefWavelength - reference wavelength (nm)
% Only specify the following parameters if the simulation requires it
%  .Att - fibre attenuation (dB/km)
%  .D - dispersion parameter at reference wavelength (ps/nm/km)
%  .S - dispersion slope at reference wavelength (ps/nm^2/km)
%  .PMD - PMD parameter at reference wavelength (ps/km^0.5)
%  .Gamma - nonlinear parameter (/W/km)
%
% Returns:
% SignalOut - output signal structure
%
% Author: Benn Thomsen, 16 June 2005.
%
% see also NLSE
%

c=3e5;              % nm/ps

[Np,Nt] = size(Signal.Et);                  % Total number of points
dF = 1e-12*Signal.Fs/Nt;                    % Spectral resolution (THz)
FF = [0:(Nt/2)-1,-Nt/2:-1] * dF;            % Frequency array (THz)

if ~isfield(P, 'Gamma') && ~isfield(P, 'PMD'),
    Nz=1;
    dz=P.Length;
else
    Nz=ceil(P.Length/P.dz);
    dz=P.Length/Nz;
end 

if isfield(P, 'Att')
    a=P.Att*log(10)/10;         % W/km
    %% Calculate effective length for nonlinear interaction in presence of fibre loss
    dzEff = (1-exp(-a*dz))/a;
else
    a=0;
    dzEff = dz;
end

if isfield(P, 'D')
    B2=-P.D*P.RefWavelength.^2/(2*pi*c);   	                        % ps^2/km
    d2=1i/2*B2*(2*pi*FF).^2;
else
    d2=0;
end

if isfield(P, 'S')
    B3=P.RefWavelength.^2/(2*pi*c).^2*(P.RefWavelength.^2*P.S+2*P.RefWavelength*P.D);      % ps^3/km
    d3=1i*B3/6*(2*pi*FF).^3;
else
    d3=0;
end

%% Dispersion operator
D = ones(Np,1)*exp(dz/2*(d2+d3)); 

if isfield(P, 'PMD')
    mean_DGD = P.PMD*sqrt(P.Length);                        % mean DGD (ps)
    DGD_dz = mean_DGD/(0.9213*sqrt(Nz));
    %% 1st order PMD operator
    PMD =exp(-1i*(DGD_dz/4)*(2*pi*FF));
    D = [PMD; 1./PMD].*D;
end

if isfield(P, 'Gamma'),
    Po=max(max(abs(Signal.Et).^2));
    Lnl=1/(P.Gamma*Po);
else
    D=D.^2;    
end
                                                         
% Ef=ifft(Signal.Et,[],2);                              % Fourier Transform to frequency domain
% if isfield(P, 'Gamma') && Np==2,
%     disp('Using Two Axis Split Step NLSE with SPM and XPM')
%     for n=1:Nz,
%         Ef=Ef.*D;                                                   % Apply Dispersion operator for the first half step
%         Et=fft(Ef,[],2);                                            % Fourier Transform to time domain
%         N=dzEff*1i*8/9*P.Gamma*[ abs(Et(1,:)).^2 + abs(Et(2,:)).^2; abs(Et(2,:)).^2 + abs(Et(1,:)).^2];  % Nonlinear operator SPM & XPM
%         Et=Et.*exp(N-dz*a/2);                                       % Apply Nonlinear and loss operators at center of step
%         Ef=ifft(Et,[],2);                                           % Fourier Transform to frequency domain   
%         Ef=Ef.*D;                                                   % Apply Dispersion operator for the second half step                    
%     end

Ef=ifft(Signal.Et,[],2);                              % Fourier Transform to frequency domain
if isfield(P, 'Gamma') && Np==2,
    %disp('Using Two Axis Split Step NLSE with SPM and XPM')
    for n=1:Nz,
        Ef=Ef.*D;                                                   % Apply Dispersion operator for the first half step
        Et=fft(Ef,[],2);                                            % Fourier Transform to time domain
        N=dzEff*1i*8/9*P.Gamma*[ abs(Et(1,:)).^2 + abs(Et(2,:)).^2; abs(Et(2,:)).^2 + abs(Et(1,:)).^2];  % Nonlinear operator SPM & XPM
        Et=Et.*exp(N-dz*a/2);                                       % Apply Nonlinear and loss operators at center of step
        Ef=ifft(Et,[],2);                                           % Fourier Transform to frequency domain   
        Ef=Ef.*D;                                                   % Apply Dispersion operator for the second half step                    
    end
else
    disp('Simulating Dispersion only')
    Ef=Ef.*D.*exp(-dz*a/2);                                          % Apply Dispersion and loss operators only
end
    
Signal.Et=fft(Ef,[],2);                              % Fourier Transform to time domain

