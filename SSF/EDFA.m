function Signal=EDFA(Signal,P,len,loss)

% Black box EDFA models gain, gain saturation and ASE
% The noise is split equally to both polarisation states
% Uses 'fsolve' from the optimisation toolbox
% Signal = EDFA(Signal,P)
%
% Inputs:
% Signal - input signal structure
% P.GdB  - amplifier gain (dB)
% P.NFdB - amplifier noise figure (dB)
% P.PsatdBm - saturated output Power (dBm) at G=Go/2 (Go small signal gain) [optional]
%
% Returns:
% SignalOut - output signal structure 
%
% Author: Benn Thomsen, May 2005.

    
h=6.62606957e-34;            % Plank constant (J.s)
[Np,Nt]=size(Signal.Et);

if isfield(P, 'PsatdBm'),
    Go=10^(P.GdB/10);                                               % Linear small signal gain
    PsatOut = 10^((P.PsatdBm-30)/10);                               % Linear saturated output Power at G=Go/2 (W) 
    Psat = (Go-2)/(Go*log(2))*PsatOut;                              % Saturated output Power (W)
    Pin = sum(sum(abs(Signal.Et).^2,1),2)/Nt;                       % Average Signal Power (W)
    GainSat = @(G,Go,Pin,Psat) Go*exp(-(G-1)*Pin/Psat)-G;           % Amplifier gain saturation (function)
    
    if(isfield(P,'verbose')&&(P.verbose>1))
        G = fsolve(GainSat,Go,optimset('fsolve'),Go,Pin,Psat);   % Determine input Power dependent gain
        fprintf('Computed EDFA gain %f dB\n',G)
    else
        options = optimset(optimset('fsolve'),'Display','off');
        G = fsolve(GainSat,Go,options,Go,Pin,Psat);              % Determine input Power dependent gain
    end


else
    G=10^(P.GdB/10);                                                % Linear small signal gain
end

if isfield(P, 'NFdB')&&(P.GdB>0),
    if Np == 1,            % check number of polarisation states if only 1 then create other state
        Signal.Et=[Signal.Et; zeros(1,Nt)];
    end
    
    NF=10^(P.NFdB/10);                  % Linear noise figure
    Nsp=(NF*G)/(2*(G-1));               % Calculate spontaneous emission factor from gain and noise figure
    Pase=2*Nsp*(G-1)*h*Signal.Fc*Signal.Fs;     % ASE Power over simulation bandwidth (W)
    
    % White Gaussian noise zero mean and standard deviation equal to ASE
    % Power split equally across the dimensions
    noiset = sqrt(0.25*Pase)*(randn(2,Nt)+1i*randn(2,Nt));
    Signal.Et = sqrt(G)*Signal.Et+noiset;
    Signal.Et = (1/sqrt(10^((P.GdB - loss*len)/10)))*Signal.Et;  
    %Signal.Et = (1/sqrt(10^((6)/10)))*Signal.Et;  
else
    if isfield(P,'verbose')&&(P.verbose>0); disp('Gain only EDFA (no noise added)'), end
    Signal.Et = sqrt(G)*Signal.Et;
end