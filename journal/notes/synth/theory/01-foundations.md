
---

**Definition (Genus-$g$ surface).**  A compact orientable surface of genus $g \geq 0$: the sphere ($g = 0$), the torus ($g = 1$), or the connected sum of $g$ tori ($g \geq 2$). Genus counts the number of holes. 

**Definition (The body).** The body $B \subset \mathbb{R}^3$ is a [[dieudonne-modern-analysis#Compact (book p.57–58, §3.16)|compact]], [[dieudonne-modern-analysis#Connected (book p.67, §3.19)|connected]], [[lee-smooth-manifolds#Topological n-manifold with boundary (book p.25-27, §1)|topological 3-manifold with boundary]] such that $\operatorname{int} B$ is a non-empty [[ball-approx-domain#Definition — domain of class C⁰ (§1, eq. 1.1, p.2)|domain of class]] $C^0$.

---

**Proposition (Outer boundary topology).** The outer boundary $\partial B^*$ of $B$ is a compact, orientable, $C^0$ surface of genus $g \geq 0$.

*Proof.* 
1. $\partial B$ is a closed subset of $B$ and a [[lee-smooth-manifolds#Topological n-manifold with boundary (book p.25-27, §1)| topological 2-manifold without boundary]].
2. The [[lee-smooth-manifolds#Connected components (book p.607, Prop A.39)|components]] of $\partial B$ are [[lee-smooth-manifolds#Connectivity of topological manifolds (book p.8, Prop 1.11)|open]] and [[lee-smooth-manifolds#Connected components (book p.607, Prop A.39)|closed]] in $\partial B$, each a [[lee-smooth-manifolds#Open subsets of topological manifolds (book p.4, §1)|connected topological 2-manifold]].
3. Since $B$ is compact, $\mathbb{R}^3 \setminus B$ has a unique unbounded component. Let $\partial B^*$ be the component of $\partial B$ that borders it — the **outer boundary**.
4. $\partial B^*$ is compact: Since it is closed in $\partial B$ (step 2), and $\partial B$ is closed in $\mathbb{R}^3$ (since $B$ is compact hence closed); it is also bounded since $B$ is bounded. By [[dieudonne-modern-analysis#Heine-Borel in R^n (book p.63, §3.17.6)|Heine–Borel]], $\partial B^*$ is compact.
5. $\partial B^*$ is a [[massey-algebraic-topology#Surface & torus (book p.5)|surface]]: Since it is connected (by choice in step 3) and a 2-manifold without boundary (step 1). 
6. By [[massey-algebraic-topology#Classification of compact surfaces (book p.9)|classification]], $\partial B^*$ is homeomorphic to a sphere, a connected sum of tori, or a connected sum of projective planes. 
7. No closed subset of $\mathbb{R}³$ is homeomorphic to a non-orientable 2-manifold ([[massey-algebraic-topology#Models of nonorientable surfaces in Euclidean 3-space (book p.32)|Massey, 1967]]). The projective plane is non-orientable ([[massey-algebraic-topology#Projective plane is nonorientable (book p.7)|Massey, 1967]]), and any connected sum with a non-orientable summand is non-orientable ([[massey-algebraic-topology#Connected sum orientability (book p.9)|Massey, 1967]]). Therefore $\partial B^*$ homeomorphic to a sphere or a connected sum of tori by exclusion.
8. $\partial B^*$ is compact and homeomorphic to a genus-g surface and hence orientable ([[massey-algebraic-topology#Orientability and homeomorphic (book p.26)|Massey, 1967]])
9. $\partial B^*$ is the class-$C^0$ (local-graph) boundary assumed of $B$ [[ball-approx-domain#Definition — domain of class C⁰ (§1, eq. 1.1, p.2)|(Ball & Zarnescu, 2017)]]
10. Therefore $\partial B^*$ is a compact orientable $C^0$ surface with the topology of a sphere ($g = 0$) or the connected sum of $g \geq 1$ tori.   $\square$

The bounded region enclosed by $\partial B^*$ — the "inside" $D_1$ of the [[zbmath#Jordan-Brouwer separation theorem|Jordan–Brouwer Separation Theorem]]— is the **optically observable hull** $B^*=D_1 \cup \partial B^*$. Light reflects off $\partial B^*$ and exits into the unbounded exterior; the cavities of $B$ are optically inaccessible.


---

**Proposition (Smooth surfaces are dense in the continuous).**  $\partial B^*$ can be approximated arbitrarily close by a smooth surface of the same topology. Formally, for every $\varepsilon>0$ there exists a compact $C^\infty$ embedded surface $S\subset\mathbb{R}^3$, **homeomorphic to $\partial B^*$** (hence of the same genus $g$), with Hausdorff distance $d_H(\partial B^*,S)<\varepsilon$.  $S$ may be taken interior or exterior to $B^*$.

*Proof (corollary of [[ball-approx-domain#Theorem 5.1 — smooth topology-preserving approximation (§5, p.11)|(Ball & Zarnescu, 2017), Theorem 5.1]]).*
1. $\Omega:=\operatorname{int}B^*$ is a bounded **domain of class $C^0$** — it is open, connected, and its boundary $\partial B^*$ is the class-$C^0$ (local-graph) boundary assumed of $B$. 
2. Let $\rho$ be the **regularized signed distance** to $\partial B^*$ (Ball–Zarnescu Prop. 3.1, after Lieberman): $\rho$ is continuous on $\mathbb{R}^3$, smooth off $\partial B^*$, and $\{\rho=0\}=\partial B^*$. For $\varepsilon'\in\mathbb{R}$ set $\Omega_{\varepsilon'}=\{x\in\mathbb{R}^3 \ : \ \rho>\varepsilon'\}$.
3. Theorem 5.1 gives $\varepsilon_0=\varepsilon_0(\Omega)>0$ such that for $0<|\varepsilon'|<\varepsilon_0$, $\Omega_{\varepsilon'}$ is a **bounded $C^\infty$ domain**; its boundary is the level set $\partial\Omega_{\varepsilon'}=\{\rho=\varepsilon'\}$, a **compact $C^\infty$ embedded $2$-manifold** (the boundary of a bounded smooth domain).
4. **Same topology.** Theorem 5.1 also gives a homeomorphism $f(\varepsilon',\cdot)$ of $\mathbb{R}^3$ that carries $\partial B^*$ onto $\partial\Omega_{\varepsilon'}$ and is the identity outside the band $\{|\rho|\le 3|\varepsilon'|\}$. Restricting it to $\partial B^*$ gives a homeomorphism $\partial B^*\xrightarrow{\ \sim\ }\partial\Omega_{\varepsilon'}$; since $\partial B^*$ is connected (Prop. above) so is $\partial\Omega_{\varepsilon'}$ — a *surface* — and homeomorphic closed surfaces have the same genus $g$.
5. **Hausdorff convergence.** $\partial\Omega_{\varepsilon'}=\{\rho=\varepsilon'\}$ lies in the band $\{|\rho|\le 3|\varepsilon'|\}$, an open neighbourhood of $\partial B^*=\{\rho=0\}$ that (since $\rho$ is continuous, vanishes exactly on $\partial B^*$, and $\partial B^*$ is compact) shrinks to $\partial B^*$ as $\varepsilon'\to0$. The approximating boundaries therefore converge to $\partial B^*$ in Hausdorff distance, $d_H(\partial B^*,\partial\Omega_{\varepsilon'})\to0$ — exactly the Lebesgue + **Hausdorff** convergence stated for these regularized-distance approximations in [[carlo-smooth-approx#Main result (abstract / §1)|Antonini (2023)]] (and carried by the band of Theorem 5.1). 
6. Given $\varepsilon>0$, choose $\varepsilon'$ with $d_H<\varepsilon$ and set $S:=\partial\Omega_{\varepsilon'}$. The two signs of $\varepsilon'$ give an interior and an exterior approximant (Theorem 5.1). $\square$

The fantastic thing about [[ball-approx-domain#Definition — domain of class C⁰ (§1, eq. 1.1, p.2)|(Ball & Zarnescu, 2017)]] is that it provides an extensive tool-set for tackling the below conjectures together with scale-space theory. Among other the paper defines the pseudo-normal and how you can construct the smooth approximation.

We henceforth write $\partial B^\infty$ as the arbitrarily close smooth surface approximation of $\partial B^*$ with the same topology. Approximated sufficiently close to be indistinguishable from $\partial B^*$. The meaning of indistinguishable formulated in the conjectures below. 


---

