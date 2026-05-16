#include "nearest_point.h"

#ifdef _DEBUG
  #undef _DEBUG
  #include <vector>
  #include <algorithm>
  #include <numeric>
  #define _DEBUG
#else
  #include <vector>
  #include <algorithm>
  #include <numeric>
#endif

struct Parabola {
    // center (minimum) of the parabola
    int32 center;

    // left-most point for which parabola is part of the lower envelope
    int32 lbound;

    // value of the parabola at the center
    int32 value;

    // index of source point for this parabola
    int32 index;
};

class NearestPointImageComputer {

  public:
    void compute(BBox bbox, int32 n_points, const Point* points, int32* image,
                 double* dist_retval = nullptr);
    void compute1(int32 n_points, const int32* points, const int32* p_tags,
                  int32 width, int32 stride, int32* f_values, int32* f_tags);
    void compute2(int32 n, int32* f_values, int32* f_tags, int32 max_val);

  private:
    std::vector<Parabola> m_parabolas;
};

void
NearestPointImageComputer::compute(
    BBox bbox, int32 n_points, const Point* points, int32* image, double* dist_retval)
{
    std::vector<int32> indices(n_points);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [=](int32 i, int32 j) {
        const Point& a(points[i]);
        const Point& b(points[j]);
        return a.x == b.x ? a.y < b.y : a.x < b.x;
    });

    const int32 x0 = bbox.ll.x;
    const int32 y0 = bbox.ll.y;
    
    const int32 w = bbox.ur.x - x0;
    const int32 h = bbox.ur.y - y0;

    // worst case is one parabola per point
    m_parabolas.resize(w);
    
    std::vector<int32> f_values(w*h, w*w+h*h);
    std::vector<int32> y_values;

    int32 i = 0;
    while (i != n_points) {

        const int32 x = points[indices[i]].x;
        
        int32 j = i+1;
        while (j != n_points && points[indices[j]].x == x)
            ++j;

        if (j-i > y_values.size())
            y_values.resize(j-i);

        for (int k = i; k != j; ++k)
            y_values[k-i] = points[indices[k]].y - y0;

        compute1(j-i, y_values.data(), indices.data()+i,
                 h, w, f_values.data() + x - x0, image + x - x0);

        i = j;
    }

    for (int i = 0; i != h; ++i) {
        compute2(w, f_values.data() + i*w, image+i*w, w*w+h*h);
    }

    if (dist_retval) {
        for (int i = 0; i != w*h; ++i)
            dist_retval[i] = sqrt(f_values[i]);
    }
}

void
NearestPointImageComputer::compute1(
    int32 n_points, const int32* points, const int32* p_tags,
    int32 width, int32 stride, int32* f_values, int32* f_tags)
{
    int32 x = 0;
    for (int32 i = 0; i != n_points; ++i) {
        
        const int32 curr_point = points[i];
        const int32 curr_tag = p_tags[i];

        int32 s = width;
        if (i+1 < n_points)
            s = std::min((curr_point + points[i+1] + 1) >> 1, width);
        
        while (x < s) {
            const int32 d = x - curr_point;
            f_values[x * stride] = d*d;
            f_tags[x * stride] = curr_tag;
            ++x;
        }
    }
}

void
NearestPointImageComputer::compute2(
    int32 n, int32* f_values, int32* f_tags, int32 max_val)
{
    int k = 0; // number of parabolas

    for (int i = 0; i != n; ++i) {

        if (f_values[i] == max_val)
            continue;

        int32 s = 0;
        while (k > 0) {

            const auto p = m_parabolas.data() + k-1;

            const int32 j = p->center;

            // we're more than rounding up, the interval [x,x+1) will map to x+1
            //
            // this means that in case of a tie (parabola p and the new
            // parabola we're considering) would yield the same value at a
            // particular coordinate then we leave p with ownership of that
            // coordinate

            // s = ((f_values[i] + i*i) - (p->value + j*j)) / (2 * (i-j)) + 1;
            
            s = (f_values[i] + i*i) - (p->value + j*j);

            // integer arithmetic rounds towards 0, not towards smaller
            // numbers, so if the result of a division is -1/2 we can end up
            // with the final value of s being 1 instead of 0.  Oops.
            s = (s < 0) ? 0 : (s / (2*(i-j)) + 1);
                
            if (s > p->lbound)
                break;
            --k;
        }

        if (s < n) {
            m_parabolas[k] = Parabola{.center=i, .lbound=s, .value=f_values[i], .index=f_tags[i]};
            ++k;
        }
    }

    if (k == 0)
        return;

    auto p = m_parabolas.data();
    const auto p_end = p + k;
    for (int32 i = 0; i != n; ++i) {

        while (p+1 != p_end && p[1].lbound <= i)
            ++p;

        const int d = i - p->center;
        f_values[i] = p->value + d * d;
        f_tags[i] = p->index;
    }
}

void
compute_nearest_point_image(
    BBox         bbox,
    int32        n_points,
    const Point* points,
    int32*       image_retval,
    double*      dist_retval)
{
    NearestPointImageComputer npic;
    npic.compute(bbox, n_points, points, image_retval, dist_retval);
}
    
