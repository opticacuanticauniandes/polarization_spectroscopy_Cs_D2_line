# polarization_spectroscopy_Cs_D2_line
Calculates the polarization spectroscopy signal for the D2 line in Cs using Rate Equations.



The python scripts:

OBEv4 calculates, given a value of detuning delta and pump polarization q and other laser/atom parameters the evolution of populations in time using the Rate Equations (derived from the Optical Bloch Equations). It returns the graph of the interaction time weighting function H(t), and the integral of the populations*H(t).



OBEv5 gives the Polarization Spectroscopy signal by calculating the integral given in OBEv4 for the values of delta (detuning of the probe and pump beam) in a desired range.
