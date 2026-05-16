#ifndef NEAREST_POINT_H
#define NEAREST_POINT_H

typedef int int32;

struct Point {
    int32 x, y;
};

struct BBox {
    Point ll, ur;
};

void compute_nearest_point_image(
    BBox         bbox,
    int32        n_points,
    const Point* points,
    int32*       image_retval,
    double*      dist_retval = nullptr);

#endif
