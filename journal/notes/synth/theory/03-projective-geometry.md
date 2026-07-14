### Projective geometry

#### Central projection (perspective/pinhole camera model)
A scene point $\mathbf{p} = (p_x, p_y, p_z)^\top$ is imaged by drawing the line through the **camera centre** $c$ and $\mathbf{p}$; the image point is where that line intersects the **image plane** $p_z = f$. By similar triangles, $\mathbf{p} \mapsto (f p_x/p_z,\ f p_y/p_z)^\top$. In homogeneous coordinates $\mathbf{x} = \Pi\mathbf{P}$ with camera matrix $\Pi = \operatorname{diag}(f,f,1)[\mathbf{I}\mid\mathbf{0}]$. The perpendicular from $c$ to the image plane is the **principal axis**; its foot is the **principal point**. ([[hartley-mulview-geom#Central projection — pinhole camera model (book p.153–155, §6.1)|Hartley & Zisserman, 2004]]). The below is defined relative to the central projection.
#### Front-facing and back-facing
This is the **local, single-surface** case: one surface $S$ in the scene, enclosing a body $M\subset \mathbb{R}^3$, viewed from camera centre $c$. Front/back is a property of the surface at a point, decided in an arbitrarily small neighbourhood of that point; occlusion by other parts of the scene plays no role here — it enters later, when $\Gamma$ is projected to the visible apparent contour $\gamma$.

**$C^0$ definition (no normal required).** A surface point $\mathbf{p}$ is **front-facing** if there is an $\varepsilon>0$ such that $\mathbf{p}+s\,(c-\mathbf{p})$ lies outside $M$ for every $s\in(0,\varepsilon)$; **back-facing** if there is an $\varepsilon>0$ such that $\mathbf{p}+s\,(c-\mathbf{p})$ lies inside $M$ for every $s\in(0,\varepsilon)$. In words: step an infinitesimal amount from $\mathbf{p}$ toward the camera; front-facing if that step leaves the body, back-facing if it enters the body. ([[ben-her-line-draw#Front-facing and back-facing — C⁰ (paper p.25, §3.4, Def 3.4.1)|Bénard & Hertzmann, 2019]])

**$C^1$ reduction.** If $S$ is $C^1$ at $\mathbf{p}$ with outward unit normal $\mathbf{n}$, then near $\mathbf{p}$ the body is $\{x:(x-\mathbf{p})\cdot\mathbf{n}\le 0\}$ to first order, so $(\mathbf{p}+s(c-\mathbf{p})-\mathbf{p})\cdot\mathbf{n}=s\,(c-\mathbf{p})\cdot\mathbf{n}$ fixes the side of the step. The $C^0$ definition therefore reduces to: front-facing iff $(c - \mathbf{p}) \cdot \mathbf{n} > 0$, back-facing iff $(c - \mathbf{p}) \cdot \mathbf{n} < 0$. ([[ben-her-line-draw#Front-facing and back-facing — C¹ (paper p.25, §3.3)|Bénard & Hertzmann, 2019]])

#### Occluding contour generator
The **occluding contour generator** $\Gamma$ is the curve on the surface $S$ delineating the frontier between front- and back-facing regions. ([[ben-her-line-draw#Occluding contour generator — Definition 3.4.1 (paper p.25, §3.4)|Bénard & Hertzmann, 2019]]). **$C^0$ characterisation:** let $V$ be the set of front-facing points of $S$; then $\Gamma = \partial V$, the boundary of $V$ taken in $S$. **$C^1$ reduction:** if $S$ is $C^1$, then $V = \{(c-\mathbf{p})\cdot\mathbf{n}>0\}$ and its boundary is $\Gamma = \{\mathbf{p} \in S : (\mathbf{p} - c) \cdot \mathbf{n} = 0\}$ — the points where the visual ray is tangent to the surface. ([[cip-vis-mot#Contour generator and apparent contour (paper p.1106)|Cipolla, 1998]])

#### Occluding contour (apparent contour)
The **occluding contour** (or **apparent contour**) $\gamma$ is the visible 2D projection of $\Gamma$ onto the image. ([[ben-her-line-draw#Occluding contour / apparent contour — Definition 3.4.2 (paper p.26, §3.4)|Bénard & Hertzmann, 2019]])

---

**Conjecture (Smooth outer boundary occluding contour equivalence).** The apparent contours $\gamma$ of $\partial B^*$ and $\partial B^\infty$ coincide at sensor resolution.  As long as the camera center is sufficiently far from the surface; $dist(c(T), \partial B^*) > d_{min}$
	


$d_{min}> 0$ is physically guaranteed by the camera housing. Objects don't penetrate the encasing, so the camera center is always at least that distance away from any surface in the scene. We henceforth assume $∂B^*$ is $C^\infty$ without loss of observational generality. Any error this assumption brings will be below what any camera is able to sense, since for any camera, a fine enough $\partial B^\infty$ keeps the error below _that camera's_ floor.



---

