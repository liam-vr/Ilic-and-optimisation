# Special relativistic functions

## Fundamental quantities
```math
\begin{align}
    \gamma(\mathbf{v})&=\frac{1}{\sqrt{1-|\mathbf{v}|^2/c^2}} \\
    D(\mathbf{v})&=\gamma(\mathbf{v})(1-\beta_x) \\
    A^{0'}&=\gamma(\mathbf{v})(A^{0} - \boldsymbol{\beta}\cdot\mathbf{A}) \\
    \mathbf{A}'&=\mathbf{A} + \boldsymbol{\beta} \left[ \frac{\gamma^2}{\gamma+1} (\boldsymbol{\beta}\cdot \mathbf{A}) - \gamma A^{0} \right]
\end{align}
```

## Relativistic angles
```math
\begin{align}
    \sin\theta'&=\frac{1}{D} \left[-\gamma \beta_y + \frac{\gamma^2}{\gamma+1} \beta_x \beta_y  \right] \\
    \cos\theta'&=\frac{1}{D} \left[-\gamma \beta_x + 1 + \frac{\gamma^2}{\gamma+1} \beta_x^2 \right] \\
    \gamma_{\varepsilon}&=\gamma_{\mathbf{v}} \gamma_{\mathbf{u}} \left( 1 + \frac{\mathbf{v}\cdot\mathbf{u}}{c^2} \right) \\
    \sin\varepsilon&=\frac{ (\mathbf{u} \times \mathbf{v})_z }{c^2} \frac{ \gamma_{\mathbf{v}} \gamma_{\mathbf{u}} (1+\gamma_{\varepsilon}+\gamma_{\mathbf{v}}+\gamma_{\mathbf{u}}) }{(1+\gamma_{\varepsilon})(1+\gamma_{\mathbf{v}})(1+\gamma_{\mathbf{v}})} \\
    \cos\varepsilon&=\frac{ (1+\gamma_{\varepsilon}+\gamma_{\mathbf{v}}+\gamma_{\mathbf{u}})^2 }{(1+\gamma_{\varepsilon})(1+\gamma_{\mathbf{v}})(1+\gamma_{\mathbf{v}})} - 1
\end{align}
```

## Relativistic velocity addition
```math
\mathbf{v}\oplus\mathbf{u}=\frac{1}{1+\frac{\mathbf{v}\cdot\mathbf{u}}{c^2}} \left[ \mathbf{v} + \frac{\mathbf{u}}{\gamma_{\mathbf{v}}} + \frac{\gamma_{\mathbf{v}}^2}{\gamma_{\mathbf{v}}+1} \frac{\mathbf{v}\cdot\mathbf{u}}{c^2} \mathbf{v} \right]
```