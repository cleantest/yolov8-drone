***Accuracy Analysis

The YOLO classifier performs well overall, achieving ~79–80% accuracy with strong consistency across most office item categories.
The errors present mostly stem from visual/object similarity and limited data in certain classes, which can be improved through data balancing and augmentation.


***Macro -F1 Analysis

The YOLO-CLS model for the 7-class classification task shows a strong overall performance, achieving an Accuracy of 79.5% and a Macro F1-Score of 0.7891. The model performs exceptionally well on the Mouse (F1: 0.9302), Envelope (F1: 0.9268), and Sanitizer (F1: 0.8947) classes, with the mouse class notably achieving perfect precision (1.0000). 
The main area for improvement is the Stapler class, which is the most concerning with the lowest F1-Score (0.5000), mainly because of being frequently misclassified as a boxcutter (8 instances). The Boxcutter class also struggles with low recall (0.6129), meaning the model misses many true instances, despite being highly precise when it does make a prediction. Conversely, the Smartphone class has good recall but suffers from lower precision (0.6500), highlighting a problem with false positives where other objects are mistaken for a smartphone.
 To enhance the model, the focus should be on adding more diverse training images for the Stapler class to reduce its confusion with the boxcutter, and reviewing the false positive cases for the smartphone to improve precision.


 ***Confusion matrix analysis

The confusion matrix shows that the model performs strongly across all the classes, with most predictions concentrated along the diagonal- indicating high accuracy. Classes like the boxcutter, envelope, and smartphone achieved near-perfect recognition, while background, mouse, sanitizer and stapler also performed well with only few misclassifications. A few errors occurred where the mouse was occasionally confused with smartphone or sanitizer, likely due to shape or lighting similarities, and the stapler showed slight confusion with boxcutter, consistent with previous F1 score analysis. Overall, the matrix confirms that the model maintains a strong recognition capability, aligning with its reported 79–80% accuracy and balanced F1 macro score of 0.7891, with only small errors stemming from visual and lighting overlaps between certain classes.