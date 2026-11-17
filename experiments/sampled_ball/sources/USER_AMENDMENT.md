The user's current request replaces the previous continuous acquisition experiment.
Target approximately two hours including preparation, with measured ETA updates.
If the run appears unlikely to finish within two hours, the user explicitly permits
reducing N_max / the common response horizon to 5000. This permission overrides
the pasted request's prohibition against a primary horizon below 10000.
The calibration budget remains N_cal=400 and tau0 is never recalibrated.
Original continuous run is gracefully stopped, with its checkpoints retained.
