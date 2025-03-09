# eLITE_Shaders
Freemium shaders for ReShade by NiceGuy.

# eLITE Motion: High-Performance Motion Estimation

Motion estimation plays a crucial role in enhancing the visual fidelity of ReShade-powered effects, particularly for ray tracing shaders and temporal filtering techniques. eLITE Motion is a cutting-edge motion estimation shader designed to provide users with a fast and highly accurate solution for their graphics enhancement needs.

Unlike conventional motion estimation methods, eLITE Motion leverages Gradient Ascent on Normalized Cross-Correlation for block matching. This approach ensures precise motion vector calculation with minimal performance overhead. By optimizing block-matching efficiency through using Gradient Ascent, eLITE Motion outperforms most alternatives in both speed and accuracy, making it an excellent choice for users who prioritize high-quality motion estimation without compromising on performance. While on the other hand, the use of NCC mitigates the traditional issue with lighting changes being incorrectly detected as motion.

One of the primary applications of eLITE Motion is within denoising passes for RT shaders such as CompleteRT and NiceGuy Lighting. These shaders benefit significantly from accurate motion estimation, as it helps preserve details and reduce temporal artifacts in dynamic scenes. Additionally, eLITE Motion is compatible with temporal filtering shaders like TFAA, further expanding its usability within the ReShade ecosystem.
