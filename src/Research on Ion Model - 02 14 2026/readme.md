## Source Code Overview

This directory contains all active and historical project code for the AI‑Research workspace. Each project has its own subfolder and follows a consistent structure to make the codebase easy to navigate, rebuild, and analyze.

Projects may be written in **Python**, **C**, **VB.NET**, **Rust**, or other languages as needed.

Each project inside this directory will have independent **readme.md**, **status.md**, and **build.md** files.
  - **status.md** — a short, structured summary of the projects current state.
  - **build.md** — contains build instructions, envrionment notes, including required libraries, compiler or interpreter versions, known build issues, and troubleshooting notes if available.  Expect that nightmares will be described here when the MSVC linker, or build tools are involved.
   
Some projects may also include a samples subfolder containing example outputs, generated data, models, or other artifacts relevant to the research or purpose of the project.

### System Requirements and Expectations

Most of these projects assume you have a working knowledge of your operating system and development environment.

All original development was performed on:
- Windows 10
- AMD Threadripper 3960X (Zen 2)
- 128 GB of DDR4 RAM
- Three MSI 1050 Ti GPUs (4GB VRAM each)

Many projects will require modification to run on your system, especially if you are using different hardware, copilers and/or operating systems.
- **Email:** github.projects@muskokatech.club
- **Discord:** pdrifting *(Please include a clear reason for contacting me or you will be ignored.)*

### Purpose of These Projects

All projects in this directory are part of an ongoing effort to:
- reduce compute requirements for AI experimentation
- explore alternative architectures and training behaviors
- document failures, dead ends, and unexpected behavior
- identify pitfalls in current AI model design
- push toward more accessible, low‑compute AI research

This repository is intended to support reproducibility, transparency, and open exploration of AI system accessibility.
