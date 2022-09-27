$$
\begin{align}
\mathrm{PQ} &= \frac{\sum_{(p, g) \in T P} \operatorname{IoU}(p, g)}{|T P|+\frac{1}{2}|F P|+\frac{1}{2}|F N|}\\

&= \underbrace{\frac{\sum_{(p, g) \in T P} \operatorname{loU}(p, g)}{|T P|}}_{\text {segmentation quality (SQ) }} \times \underbrace{\frac{|T P|}{|T P|+\frac{1}{2}|F P|+\frac{1}{2}|F N|}}_{\text {recognition quality (RQ) }}
\end{align}
$$