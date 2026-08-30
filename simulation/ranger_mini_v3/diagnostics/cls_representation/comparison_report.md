# Representation Comparison\n\n|                       |   Mean-Pool |   DINOv3 Normalized CLS |
|:----------------------|------------:|------------------------:|
| t -> t+1              |    0.993213 |                0.979988 |
| t -> t+10             |    0.963804 |                0.909443 |
| t -> t+100            |    0.925869 |                0.815806 |
| t -> t+500            |    0.921711 |                0.804892 |
| random distant        |    0.927053 |                0.817316 |
| geometric state->goal |    0.942639 |                0.85587  |
| uniform state->goal   |    0.935069 |                0.833216 |
| uniform reward=1 %    |   99.4985   |               72.8185   |
| geometric reward=1 %  |   99.4985   |               74.8245   |
| mean feature norm     |    0.9641   |                0.907844 |