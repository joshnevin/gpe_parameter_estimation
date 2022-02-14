function SignalOut = DJI_QAM(SignalIn)
%File: DJI_QAM.m
%Version: 1
%Date: 18 February 2020
%Author: David Ives
%email: di231@cam.ac.uk

%SignalIn - structure containing 
%SignalIn.Fb %symbol rate
%SignalIn.dT %symbol time
%SignalIn.Np % number of polarisations
%SignalIn.Nb % number of symbols
%SignalIn.Ns % number of samples/symbol
%SignalIn.M  % number of bit/symbol
%SignalIn.RRC % RRC alpha
%Signal.Seed % Random 

%SignalOut - structure containing
%SignalIn 
%SignalOut.IdealSym
%SignalOut.IdealBits
%SignalOut.Symbols
%Signal.Et

% generates a 1mW signal
if ~isfield(SignalIn,'Seed')
    SignalIn.Seed=0;
end
DataStream=RandStream('mcg16807','seed',SignalIn.Seed);

SignalOut=SignalIn;

switch SignalIn.M
    case 4
        SignalOut.IdealSym=(1/sqrt(2))*[-1-1i; -1+1i; +1-1i; +1+1i;];
        SignalOut.IdealBits=[0 0; 0 1; 1 0; 1 1;];
    case 16
        SignalOut.IdealSym=(1/sqrt(10))*[-3-3i; -3+3i; +3-3i; +3+3i; 3+1i; 3-1i; 1+3i; 1-3i; -1+3i; -1-3i; -3+1i; -3-1i; 1+1i; 1-1i; -1+1i; -1-1i;];
        SignalOut.IdealBits=[0 0 0 0; 0 0 1 0; 1 0 0 0; 1 0 1 0; 1 0 1 1; 1 0 0 1; 1 1 1 0; 1 1 0 0; 0 1 1 0; 0 1 0 0; 0 0 1 1; 0 0 0 1; 1 1 1 1; 1 1 0 1; 0 1 1 1; 0 1 0 1;];    
    case 64
        b=[0 0 0; 0 0 1; 0 1 1; 0 1 0; 1 1 0; 1 1 1; 1 0 1; 1 0 0;];
        SignalOut.IdealSym=(1/sqrt(42))*[-7-7i; -7+7i;  7-7i;  7+7i; -7-5i; -7-3i; -7-1i; -7+1i; -7+3i; -7+5i;
        -5-7i; -5-5i; -5-3i; -5-1i; -5+1i; -5+3i; -5+5i; -5+7i;
        -3-7i; -3-5i; -3-3i; -3-1i; -3+1i; -3+3i; -3+5i; -3+7i;
        -1-7i; -1-5i; -1-3i; -1-1i; -1+1i; -1+3i; -1+5i; -1+7i;
         1-7i;  1-5i;  1-3i;  1-1i;  1+1i;  1+3i;  1+5i;  1+7i;
         3-7i;  3-5i;  3-3i;  3-1i;  3+1i;  3+3i;  3+5i;  3+7i;
         5-7i;  5-5i;  5-3i;  5-1i;  5+1i;  5+3i;  5+5i;  5+7i;
         7-5i;  7-3i;  7-1i;  7+1i;  7+3i;  7+5i;];
        SignalOut.IdealBits=[b(round((real(SignalOut.IdealSym)*sqrt(42)+9)/2,0),:)  b(round((imag(SignalOut.IdealSym)*sqrt(42)+9)/2,0),:)] ;                
    otherwise
        error('Unknown modulation format')
end

SignalOut.Symbols=floor(rand(DataStream,2,SignalIn.Nb)*SignalIn.M)+1;


% ideal Nyquist filter in frequency domain
H=zeros(1,SignalIn.Nb*SignalIn.Ns);
H(1:SignalIn.Nb/2)=1; % for all filters
switch SignalIn.RRC
    case 0
        H(SignalIn.Nb/2+1)=sqrt(0.5); % RRC 0 , Nyquist filter
    case 0.25
        H(round((3*SignalIn.Nb)/8)+1:round((5*SignalIn.Nb)/8)+1)=cos((0:round((SignalIn.Nb)/4))*pi*2/(SignalIn.Nb)); %% RRC 0.25
    case 0.125
        H(round((7*SignalIn.Nb)/16)+1:round((9*SignalIn.Nb)/16)+1)=cos((0:round((SignalIn.Nb)/8))*pi*4/(SignalIn.Nb)); %% RRC 0.125
    otherwise
        error('Unsupported RRC')
end
H(SignalIn.Nb*SignalIn.Ns-SignalIn.Nb+1:SignalIn.Nb*SignalIn.Ns)=H(SignalIn.Nb+1:-1:2);
H=ones(SignalIn.Np,1)*H;

% expand symbols to Ns samples
SignalOut.Et=kron(SignalOut.IdealSym(SignalOut.Symbols),[SignalIn.Ns, zeros(1,SignalIn.Ns-1)]);

SignalOut.Et=fft(ifft(SignalOut.Et,[],2).*H,[],2)/sqrt(SignalIn.Np);


end