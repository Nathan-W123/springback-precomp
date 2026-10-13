Verified (read):

* R. Padmanabhan, M. C. Oliveira, A. J. Baptista, L. F. Menezes, J. L. Alves,
  "Study on the influence of the refinement of a 3-D finite element mesh in
  springback evaluation of plane-strain channel sections", AIP Conf. Proc. 908
  (NUMIFORM 2007) - abstract: solid elements are recommended for springback
  when tool radius / thickness < 5-6 (here 4 mm / 1 mm = 4), and "the in-plane
  to thickness FE size ratio is more relevant than the number of FE layers
  through-thickness". https://www.osti.gov/etdeweb/biblio/21061767
* This repository: `docs/forming.md` sections 5-7 (costs, the standard Hex8's
  bending stiffness, the explicit step, its kernel covering only the standard
  2 x 2 x 2 Hex8 and the generic dispatch at ~35 times the cost, the smoke
  study's 3.5 % springback difference at 7.1 m/s); `docs/verification.md` (removed in the cleanup; at commit 7d571e6)
  section 27 (one IM layer with 5 / 7 points within 3.5 % / 0.9 % of beam
  theory on a bent strip) and "What is not covered" ("the springback of the
  SPIF cases is not mesh-converged").

Context only (not read in full; from search results):

* C. Henrard et al., "Forming forces in single point incremental forming:
  prediction by finite element simulations, validation and sensitivity",
  Computational Mechanics 47 (2011) 573-590 - SPIF force predictions compared
  across FE codes, element types and constitutive laws.
  https://link.springer.com/article/10.1007/s00466-010-0563-4
* I. A. Burchitz, "Improvement of springback prediction in sheet metal
  forming", PhD thesis, University of Twente (2008) - cited in search results
  for elements spanning 5-10 deg of the tool radius (0.35-0.7 mm on a 4 mm
  tool); the PDF text could not be extracted, so this is unverified.
  https://ris.utwente.nl/ws/files/6069581/thesis_Burchitz.pdf
* Machina Labs, RoboForming: two robot arms on either side of a framed sheet,
  with force / displacement sensing - the process this model stands for.
  https://machinalabs.ai/resources/advanced-manufacturing-incremental-sheet-metal-forming-with-robotics-and-ai
