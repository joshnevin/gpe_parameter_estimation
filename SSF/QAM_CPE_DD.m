function [SignalOut, varargout] = QAM_CPE_DD(SignalIn, P)

%File: QAM_CPE_DD.m
%Version: 1
%Date: 29 May 2019
%Author: David Ives
%email: di231@cam.ac.uk
%License:
%
%decision directed carrier phase estimation for QAM modulation formats.
%
%input SignalIn a signal structre
%required P.ModFormat a string 'QPSK', '4QAM', '16QAM'...
%required  P.CPELength, full length of rolling window should be odd! or will be made odd
%optional P.CommonPhase averages x and y phase.
%output SignalOut a signal structure , carrier phase corrected.
%output structure with PhaseX and PhaseY the recovered phase.


%%

if ~isfield(P,'ModFormat'), error('Modulation Format must be specified in P.ModFormat'); end

% set up ideal symbols
switch P.ModFormat
    case 'QPSK'
        Sym= reshape(([-1 1]+1i*[-1; 1;])/sqrt(2),4,1);
    case '4QAM'
        Sym= reshape(([-1 1]+1i*[-1; 1;])/sqrt(2),4,1);
    case '16QAM'
        Sym= reshape(([-3 -1 1 3]+1i*[-3; -1; 1; 3;])/sqrt(10),16,1);
    case '64QAM'
        Sym= reshape(([-7 -5 -3 -1 1 3 5 7]+1i*[-7; -5; -3; -1; 1; 3; 5; 7;])/sqrt(42),64,1);
    otherwise
        error([P.ModFormat ' unrecognised modulation format'])
end

SignalOut=SignalIn;

Nt=size(SignalIn.Et,2);
halfwidth = floor(P.CPELength/2);
cut=find(SignalIn.Et(1,:)~=0,1)-1;
phase=zeros(2,Nt);
Sym=Sym*ones(1,2*halfwidth+1);

for i=halfwidth+1+cut:Nt-halfwidth-cut
    
    tempEt=SignalIn.Et(:,(i-halfwidth):(i+halfwidth)).*exp(-1i*phase(:,i-1)); % correct batch with current phase value
    SymDec(1,:)=Sym(abs(Sym-tempEt(1,:)).^2==min(abs(Sym-tempEt(1,:)).^2)).'; % decide closest symbols
    SymDec(2,:)=Sym(abs(Sym-tempEt(2,:)).^2==min(abs(Sym-tempEt(2,:)).^2)).';

    phase(:,i)=phase(:,i-1)+angle(sum(tempEt.*conj(SymDec),2)); % accumulate average phase change
    
end

% joint xy phase estimation (average x and y but leave average of x and average of y the same.
if isfield(P, 'CommonPhase')
    if P.CommonPhase
        phase(:,halfwidth+1+cut:Nt-halfwidth-cut)=mean(phase(:,halfwidth+1+cut:Nt-halfwidth-cut),2)-mean(mean(phase(:,halfwidth+1+cut:Nt-halfwidth-cut)))+mean(phase(:,halfwidth+1+cut:Nt-halfwidth-cut));
    end
end

% apply phase correction
SignalOut.Et = SignalOut.Et.*exp(-1i*phase);

o.PhaseX=phase(1,:);
o.PhaseY=phase(2,:);
varargout{1}=o;

end
    
















