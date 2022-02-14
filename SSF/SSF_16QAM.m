%File: SSF_16QAM.m
%Version: 1
%Date: 18 February 2020
%Author: David Ives
%email: di231@cam.ac.uk

tic

% GPU_Enabled=(gpuDeviceCount>0);
GPU_Enabled = false;
if GPU_Enabled, gpuDevice(); end 

%%************************************************************************
%% Simulation Parameters                                        
% Ps=[0.00025119, 0.00031623, 0.00039811, 0.00050119, 0.00063096, 0.00079433, 0.001, 0.00125893, 0.00158489, ...
%     0.00199526, 0.00251189, 0.00316228, 0.00398107];                        % launch power (W)(-6 -> +6 dBm, step of 1 dBm)
P = 1e-3*(10^(5/10));  
%Signal Parameters
Signal.Nb=2^17;    				%Number Symbols, (multiples of 421)
Signal.Np=2;           			%number of polarisations in PDM
Signal.Ns=4; 					%Number of samples per bit
Signal.Fb=  11.5e9; %  32e9;                 %Baud (Bd) - maybe try 32
Signal.Fc=193.4e12;    			 % carrier frequence (Hz)  
Signal.Fs=Signal.Fb*Signal.Ns;  %frequency span (Hz)
Signal.dT=1/Signal.Fs;          %time step (ps)
Signal.dF=Signal.Fb/Signal.Nb;  % frequency step (Hz)
Signal.Gr=50e9;                 % grid spacing (Hz)
Signal.M=4;
Signal.RRC=0.125;  % suggested by David
Signal.Seed=1;                  %randi(1000);

%Fibre System Parameters 
SMF.c=299792.458;               %[nm/ps]
SMF.Length=100;                 % km
SMF.RefWavelength=1550.116;     % nm
SMF.Att= 0.2;  %0.203;                   % dB/km
%SMF.D=16.462;                  % ps/nm/km
SMF.D=17;   %15;  
%SMF.B2=SMF.D*SMF.RefWavelength.^2/(2*pi*SMF.c); % ps^2/km
SMF.Gamma=1.2;   %1.5;                  % /W/km
SMF.NSpan=10; 
SMF.dz=0.1;                     % km

%Amplifier Characteristics
AMP.GdB=25;                     % dB
AMP.NFdB=4.6;                 % dB

%% Transmitter
Signal=DJI_QAM(Signal);
Signal.Et=sqrt(P)*Signal.Et;


figure(1)
OSA2=[];
OSA2.RB=0.10005e9;
OSA2=DJI_OSA2(Signal,OSA2);
OSA(1)=OSA2;

%% Fibre transmission
toc
for i=1:SMF.NSpan
    if GPU_Enabled
        SignalSx=Manakov_GPU(Signal,SMF);
    else
        SignalSx=Manakov(Signal,SMF);    
    end
	Signal=EDFA(SignalSx,AMP,SMF.Length,SMF.Att); % ideal gain to compensate link	
    toc
end

figure(2)
OSA2=[];
OSA2.RB=0.10005e9;
OSA2=DJI_OSA2(Signal,OSA2);
OSA(2)=OSA2;

%% Receiver
% ideal Nyquist filter in frequency domain
H=zeros(1,Signal.Nb*Signal.Ns);
H(1:Signal.Nb/2)=1; % for all filters
switch Signal.RRC
    case 0
        H(Signal.Nb/2+1)=sqrt(0.5); % RRC 0 , Nyquist filter
    case 0.25
        H(round((3*Signal.Nb)/8)+1:round((5*Signal.Nb)/8)+1)=cos((0:round((Signal.Nb)/4))*pi*2/(Signal.Nb)); %% RRC 0.25
    case 0.125
        H(round((7*Signal.Nb)/16)+1:round((9*Signal.Nb)/16)+1)=cos((0:round((Signal.Nb)/8))*pi*4/(Signal.Nb)); %% RRC 0.125
    otherwise
        error('Unsupported RRC')
end
H(Signal.Nb*Signal.Ns-Signal.Nb+1:Signal.Nb*Signal.Ns)=H(Signal.Nb+1:-1:2);
H=ones(Signal.Np,1)*H;



%% Post CD compensation
B2=-SMF.NSpan*SMF.Length*SMF.D*SMF.RefWavelength.^2/(2*pi*SMF.c); %[ps^2]
FF=[0:(Signal.Nb*Signal.Ns)/2-1,-(Signal.Nb*Signal.Ns)/2:-1] * Signal.dF/10^12; % [THz]
Dpost=ones(Signal.Np,1)*exp(1i/2*-B2*(2*pi*FF).^2);

SignalRecovered=Signal;
SignalRx.Et=fft(ifft(Signal.Et,[],2).*H.*Dpost,[],2);
SignalRecovered.Et=SignalRx.Et(:,1:4:end);

figure(3)
plot(SignalRecovered.Et.','.')
axis square

%%

% size(SignalRecovered.IdealSym(SignalRecovered.Symbols(:,501:end-500)))
% 
% size(SignalRecovered.IdealSym(SignalRecovered.Symbols(:,501:end-500))')
% 
% size(SignalRecovered.IdealSym(SignalRecovered.Symbols(:,501:end-500))*SignalRecovered.IdealSym(SignalRecovered.Symbols(:,501:end-500))')
% 
% size(SignalRecovered.Et(:,501:end-500))
% 
% size(SignalRecovered.IdealSym(SignalRecovered.Symbols(:,501:end-500))')

%%

alpha= diag(real((  SignalRecovered.IdealSym(SignalRecovered.Symbols(:,501:end-500))*SignalRecovered.IdealSym(SignalRecovered.Symbols(:,501:end-500))' ))./(SignalRecovered.Et(:,501:end-500)*SignalRecovered.IdealSym(SignalRecovered.Symbols(:,501:end-500))'));
SignalRecovered.Et=alpha.*SignalRecovered.Et;
EVM=SignalRecovered.Et(:,501:end-500)-SignalRecovered.IdealSym(SignalRecovered.Symbols(:,501:end-500));
mean(diag(-10*log10(EVM*EVM'/size(EVM,2))))

figure(4)
plot(SignalRecovered.Et.','.')
axis square

%%

p.ModFormat = '4QAM';
p.CPELength = 32;
SignalRR=QAM_CPE_DD(SignalRecovered, p);

alpha= diag(real((SignalRR.IdealSym(SignalRR.Symbols(:,501:end-500))*SignalRR.IdealSym(SignalRR.Symbols(:,501:end-500))'))./(SignalRR.Et(:,501:end-500)*SignalRR.IdealSym(SignalRR.Symbols(:,501:end-500))'));
SignalRR.Et=alpha.*SignalRR.Et;
EVM=SignalRR.Et(:,501:end-500)-SignalRR.IdealSym(SignalRR.Symbols(:,501:end-500));

mean(diag(-10*log10(EVM*EVM'/size(EVM,2))))

figure(5)
plot(SignalRR.Et.','.')
axis square

% Et=SignalRecovered.Et.*conj(SignalRecovered.IdealSym(SignalRecovered.Symbols))./abs(SignalRecovered.IdealSym(SignalRecovered.Symbols));
% plot(Et,'.')
% 
% x1s=std(real(Et(abs(SignalRecovered.IdealSym(SignalRecovered.Symbols))<0.6)));
% y1s=std(imag(Et(abs(SignalRecovered.IdealSym(SignalRecovered.Symbols))<0.6)));
% 
% x2s=std(real(Et(abs(SignalRecovered.IdealSym(SignalRecovered.Symbols))>0.6 & abs(SignalRecovered.IdealSym(SignalRecovered.Symbols))<1.2)));
% y2s=std(imag(Et(abs(SignalRecovered.IdealSym(SignalRecovered.Symbols))>0.6 & abs(SignalRecovered.IdealSym(SignalRecovered.Symbols))<1.2)));
% 
% x3s=std(real(Et(abs(SignalRecovered.IdealSym(SignalRecovered.Symbols))>1.2)));
% y3s=std(imag(Et(abs(SignalRecovered.IdealSym(SignalRecovered.Symbols))>1.2)));
% 
% mean([x1s,x2s,x3s])
% sqrt(10)*([y1s, y2s, y3s]-[x1s,x2s,x3s])./sqrt([2 10 18])

%%




