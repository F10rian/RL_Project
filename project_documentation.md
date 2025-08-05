# Documentation

## Pretraining
Trained models as basline for transfer learning.

### Crossing

Trained models on a 5x5 grid with crossing but without any success.

### Plain Grid

Trained baselines on different grid sizes (i.e. 5x5, 7x7, 11x11, 15x15, 21x21) successfully. All models converged on an average reward of 0.95 but at different steps.

## Transfer Learning

Direct transfer of the 5x5 baseline to 21x21 grid. This works but is not faster than training the 21x21 baseline from scratch.

## Curriculum Learning

