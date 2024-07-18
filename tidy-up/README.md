# Tidy Up Task

The idea behind this task is that the robot is encountering a cluttered household environment where different objects are misplaced.
This problem is encapsulated by the following two commonsense questions the robot should be able to answer:

- What objects in my environment are out of place?
- What are the prototypical locations for these objects?

To answer these questions, we provide a list of 851 household objects and their associated locations.

## Data Structure

The data in the *tidy_up_data_csv* is collected in two columns:
The first column (*Object*) contains all object names.
The second column (*Locations*) contains a Python dictionary with a ranked list of possible locations.
The key in the dictionary describes the rank of the location.
The value in the dictionary is the name of the location as well as a trust value on which the ranking is based.
The trust value depends on the data source from which this location is gathered.

Example:
```ladle,"{1: ('kitchen', 1.5), 2: ('drawer', 0.5)}"```

The object *ladle* is found probably found in the *kitchen* otherwise it is found in a *drawer*.

## Data Sources

We combine five different data sources to gather the information on prototypical object locations.
For each source, we provide a single file focused on the data extraction.
Additionally, to create the ranking of the locations, we manually assign each source a specific trust value based on the reliability.

| Source         | Trust Value | Candidates | Included | Ref |
|----------------|-------------|------------|----------|-----|
| AI2Thor        | 1.00        | 118        | 118      | [1] |
| Ascent++       | 0.50        | 4309       | 131      | [2] |
| CSKG           | 0.50        | 9264       | 462      | [3] |
| Housekeep      | 1.00        | 268        | 268      | [4] |
| Microsoft COCO | 0.75        | 7          | 7        | [5] |

## References

[1] E. Kolve et al., ‘AI2-THOR: An Interactive 3D Environment for Visual AI’, 2017, arXiv. doi: 10.48550/ARXIV.1712.05474.

[2] T.-P. Nguyen, S. Razniewski, J. Romero, and G. Weikum, ‘Refined Commonsense Knowledge from Large-Scale Web Contents’, IEEE Trans. Knowl. Data Eng., pp. 1–16, 2022, doi: 10.1109/TKDE.2022.3206505.

[3] F. Ilievski, P. Szekely, and B. Zhang, ‘CSKG: The CommonSense Knowledge Graph’, in The Semantic Web, vol. 12731, R. Verborgh, K. Hose, H. Paulheim, P.-A. Champin, M. Maleshkova, O. Corcho, P. Ristoski, and M. Alam, Eds., in Lecture Notes in Computer Science, vol. 12731. , Cham: Springer International Publishing, 2021, pp. 680–696. doi: 10.1007/978-3-030-77385-4_41.

[4] Y. Kant et al., ‘Housekeep: Tidying Virtual Households Using Commonsense Reasoning’, in Computer Vision – ECCV 2022, S. Avidan, G. Brostow, M. Cissé, G. M. Farinella, and T. Hassner, Eds., in Lecture Notes in Computer Science, vol. 13699. Cham: Springer Nature Switzerland, 2022, pp. 355–373. doi: 10.1007/978-3-031-19842-7_21.

[5] T.-Y. Lin et al., ‘Microsoft COCO: Common Objects in Context’, in Proceedings of the 13th European Conference on Computer Vision (ECCV 2014), D. Fleet, T. Pajdla, B. Schiele, and T. Tuytelaars, Eds., in Lecture Notes in Computer Science, vol. 8693. Zurich, Switzerland: Springer International Publishing, 2014, pp. 740–755. doi: 10.1007/978-3-319-10602-1_48.