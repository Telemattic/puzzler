#include "nearest_point.h"

#include <algorithm>

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
        compute2(w, f_values.data() + i*w, image + i*w);
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
    int32 n, int32* f_values, int32* f_tags)
{
    if (n < 1)
        return;
    
    if (m_centers.size() < n)
        m_centers.resize(n, -1);
    int32* centers = m_centers.data();

    if (m_lbounds.size() <= n)
        m_lbounds.resize(n+1, -1);
    int32* lbounds = m_lbounds.data();

    // index of rightmost parabola in lower envelope
    int32 k = 0;
    centers[0] = 0;
    lbounds[0] = -1;
    
    for (int32 i = 1; i != n; ++i) {

        int32 s = -1;
        while (k > 0) {
            const int32 j = centers[k];
            s = ((f_values[i] + i*i) - (f_values[j] + j*j) + (i-j)) / (2 * (i-j));
            if (s > lbounds[k])
                break;
            --k;
        }
        k += 1;
        centers[k] = i;
        lbounds[k] = s;
    }
    lbounds[k+1] = n;

    const std::vector<int32> save_f_values(f_values, f_values+n);
    const std::vector<int32> save_f_tags(f_tags, f_tags+n);

    int j = 0;
    for (int i = 0; i != n; ++i) {
        while (lbounds[j+1] <= i)
            ++j;
        const int c = centers[j];
        const int d = c - i;
        f_values[i] = save_f_values[c] + d*d;
        f_tags[i] = save_f_tags[c];
    }
}
