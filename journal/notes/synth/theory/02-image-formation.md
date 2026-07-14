### Image formation
The image intensity is given by the **rendering equation** — a geometric-optics light-transport balance (emitted + scattered): ([[kajiya-rendeq#The rendering equation (SIGGRAPH 1986)|Kajiya, 1986]])
$$I(x,x') = g(x,x')\left[\,\epsilon(x,x') + \int_S \rho(x,x',x'')\,I(x',x'')\,dx''\,\right]$$


The intensity is an **integral over the surface** $S$. Local orientation dependence is carried inside $g$ and $\rho$, so evaluating $I$ needs no explicit surface normal, only $S$ as a domain of integration. But the surface normal is in the physics of $g$ and $\rho$. 
### Scale-space theory 
The bridge from the $C^0$ outer boundary to a $C^\infty$ working model: every physical measurement integrates over a finite aperture, so sub-resolution structure is unobservable and what is observed is smooth.
#### Inner scale = sensor resolution
A measured image carries structure only between an outer scale (the image extent) and an **inner scale** set by the resolution; for a digital image the inner scale is the pixel size. ([[lind-scale-space#Inner scale = sensor resolution (p.9, §1.3.4)|Lindeberg, 1994]])


---

**Conjecture (Smooth outer boundary image equivalence).**  Let $L(S, T)$ be the image of the surface $S$ taken at camera pose $T$ by a camera with finite resolution and quantized intensity.  Pose is defined relative to the surface $S$ which is fixed in space. Let $B$ and $\partial B^*$ be defined as above. Then for every standoff $d_{min}> 0$  there exists a smooth surface $\partial B^\infty$ such that $L(\partial B^*, T) = L(\partial B^\infty,T)$. As long as the camera center is sufficiently far from the surface; $dist(c(T), \partial B^*) > d_{min}$. 

So you can treat all $C^0$ surfaces as  $C^\infty$ as long as you "don't look too close with a microscope".

*Note.*
1. I believe a proper proof would have to leverage techniques from scale-space theory. 
2. Kajiyas rendering equation may also not be the right image formation formula to use. 
3. The arguments below just make the rationale for why I believe it could be true.  

*Arguments.*
1. The camera observes $\partial B^*$ only through a finite aperture, so the image is band-limited and therefore $C^\infty$ ([[lind-scale-space#Infinite differentiability — the key claim (p.45, §2.4.9)|Lindeberg, 1994]]); equivalently it is a finite-scale observation in the scale-space sense. Hence there exists a $C^\infty$ surface producing an image indistinguishable from $\partial B^*$'s at the sensor's resolution, and $\partial B^*$ may be taken $C^\infty$ without observational loss.

2. Any physical observation integrates the signal over a finite (non-infinitesimal) window: ideal point measurements can never be performed, since a detector needs finite energy for a registrable response. Consequently **structures with characteristic length below the aperture scale $\sigma$ are suppressed** — the sensor cannot resolve them. ([[lind-scale-space#Finite aperture: ideal point measurements are physically impossible — the most direct statement (pp.40–41, §2.4.3)|Lindeberg, 1994]])

Alternatively:
  1. Let $L(S,T)=I$ be the image built from Kajiyas rendering equation. 
  2. Let $(\partial B^\infty_k)_{k \in \mathbb{N}}$ be a sequence of surfaces which approach $\partial B^*$ in the limit.  
  3. Assume $L(S,T)$ is continuous in $S$. Then $L(\partial B^\infty_k, T)$ converges toward $L(\partial B^*, T)$ in the limit.
  4. Pick $k=N$ large enough such that the quantization can't capture the residual error when the distance is above the standoff and then set $\partial B^\infty = \partial B^\infty_N$

---
