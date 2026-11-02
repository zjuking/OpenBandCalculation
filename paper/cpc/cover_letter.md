# Cover letter

**To:** The Editors, *Computer Physics Communications*
**Re:** submission as a Computer Programs in Physics (CPiP) article

---

Dear Editors,

Please consider the enclosed manuscript, **"OBPDS 2.0: a verified
one-dimensional Schrödinger–Poisson and drift–diffusion solver for III–V
heterostructures"**, for publication in *Computer Physics Communications* as a
Computer Programs in Physics article.

**What the program does.** OBPDS 2.0 solves the one-dimensional electrostatic,
transport and envelope-Schrödinger problems of III–V heterostructures on a
single grid: self-consistent band diagrams, terminal currents, capacitances and
quantised subband structure, for layer stacks that may be pseudomorphically
strained. It is a modern rewrite of the Open Band Parameters Device Simulator,
which stopped at the zero-current approximation and could therefore not produce
a current at all. The rewrite adds drift–diffusion transport, self-consistent
quantisation and strain, and keeps the non-parabolic multi-valley statistics of
the original.

**Why we think it belongs in CPC.** The contribution is numerical rather than
physical, and it is the kind that is usually lost. Two examples. First,
Scharfetter–Gummel fluxes written in generalised potentials and rearranged to
be cancellation-free return an *identically* zero equilibrium current; the
textbook form returns about 10^10 A/cm² of round-off noise at the same
operating point. Second, in III–V materials the carrier densities inside one
device span more than twenty decades, so the discrete current in a
quasi-neutral region sits below the round-off of the stored quasi-Fermi levels.
The program reads the terminal current where it is clean and reports the spread
of the discrete current, so a user can see when a number is arithmetic noise.
Both are the sort of detail that decides whether a code is usable, and both are
invisible in a paper that only reports device results.

**Verification.** Every model in the program is held to closed-form results by
190 assertions, including negative controls: a classical p–n junction must come
back bit-for-bit unchanged when the quantum correction is switched on, a layer
on its own substrate must not be strained, and the equilibrium current must
vanish identically. The suite is part of the archive and runs in about four
minutes. We would rather be judged on that than on the device numbers.

**The archive.** The submission package contains the full source (about 3,100
lines of Python, 1,700 of front end), the four verification suites, a
comprehensive example run with its expected output, the scripts that regenerate
every figure and table of the manuscript, and a README covering installation
and operation. A browser front end is included for classroom use; it calls the
same public API and contains no numerics.

**Licence.** The program is released under **AGPL-3.0-or-later**. This is not a
preference: OBPDS is distributed under the AGPL, our program is a derivative of
it, and the licence therefore carries over. We note that AGPL-3.0 does not
appear on the list of approved licences in the Guide for Authors. We would be
grateful for guidance from the Technical Editor on whether the CPC Program
Library can accept it, or whether a different arrangement is preferred; we are
happy to discuss this before the submission is processed.

**Declarations.** The manuscript is original, is not under consideration
elsewhere, and all authors have approved the submission. The authors declare no
competing interests. Generative AI tools were used to draft and refactor code
and to prepare a first draft of the text; all calculations were run and checked
by the authors, and this is declared in the manuscript.

Yours sincerely,

First Author (corresponding author)
Affiliation, City, Country
author@example.edu
