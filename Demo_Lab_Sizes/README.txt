Test only: 5 µm test images renamed as D10 / D05 experiments to check the per-size post-processing.
Not real 7/10 µm data.

  G1-C0.30-F2.0-P7.5-S40000-SCA-D10 (99)/IN/PV 15/Fused_St_Filt_Comp.jpg
      = official test image Image_001 (experiment (1), G1, IN, PV 03; 5 µm)
  G1-C0.20-F0.9-P8.0-S21000-SCP-D05 (98)/IN/PV 15/Fused_St_Filt_Comp.jpg
      = official test image Image_005 (experiment (1), G1, IN, PV 15; 5 µm)

Segmenting this folder shows that the folder name decides the method: the "D10" experiment is post-processed
with circle reconstruction, the "D05" experiment with closing (docs/WINDOWS_TEST_CHECKLIST.txt, steps 5-6).
The scan lists this README.txt as skipped ("file at top level") and warns that the Before and WT folders are
missing - both are expected for this small test folder.
