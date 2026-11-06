"Contour" can mean a level curve of a function (like a topographic map) or the outline of a shape in an image. Curvature connects to both.

## Contour as a level curve

A contour is the set where a function $f(x,y)$ equals a constant $c$:

$$
f(x,y)=c
$$

The gradient $\nabla f$ is perpendicular to the contour, so the unit normal is:

$$
\mathbf{n}=\frac{\nabla f}{|\nabla f|}
$$

Curvature measures how fast that normal turns as you move along the curve, and it comes out as the divergence of the unit normal:

$$
\kappa=\nabla\cdot\left(\frac{\nabla f}{|\nabla f|}\right)
$$

Expanded, this is:

$$
\kappa=
\frac{
f_{xx}f_y^2-2f_{xy}f_xf_y+f_{yy}f_x^2
}{
(f_x^2+f_y^2)^{3/2}
}
$$

So the contour's curvature is determined entirely by the first and second derivatives of $f$. The first derivatives set the direction of the normal, and the second derivatives tell you how it changes.

## Contour as a parametrized curve

If the contour is given as $(x(t),y(t))$, then:

$$
\kappa=
\frac{x'y''-y'x''}
{(x'^2+y'^2)^{3/2}}
$$

This is the same quantity, $1/R$, where $R$ is the radius of the best-fit circle at that point. The sign tells you which way the curve bends, such as convex versus concave.

## Intuition on a topographic map

- Straight contours mean the slope runs in one constant direction, so curvature is about $0$, like a planar hillside.
- Tightly curved contours mean the terrain changes direction sharply. Contours bulging downhill mark a ridge, and contours bulging uphill mark a valley or gully.
- Closed, small loops mean a peak or pit, where curvature is large.
- Contour curvature is not the same as the surface's curvature. It describes the horizontal bending of the line. The surface also curves along the slope direction (profile curvature), and terrain analysis uses both.

## Where it gets used (ML and vision)

- **Shape analysis**: Corners and inflection points on an object's outline are where contour curvature peaks or changes sign. This is the basis of curvature scale space and many corner detectors.
- **Curve evolution**: In level-set methods and active contours, the contour moves at a speed proportional to its curvature (curvature flow). It smooths the contour, which is how image segmentation keeps boundaries from being jagged.
- **Implicit surfaces**: For a signed distance function, $|\nabla f|=1$, so the formula simplifies to $\kappa=\nabla^2 f$ (the Laplacian) in 2D.

If you had a specific context in mind, such as a course problem, image processing, or terrain, tell me and I can go deeper or work an example.
