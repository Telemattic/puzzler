#ifndef NEAREST_POINT_H
#define NEAREST_POINT_H

#ifdef _DEBUG
  #undef _DEBUG
  #include <vector>
  #include <algorithm>
  #include <memory>
  #include <numeric>
  #define _DEBUG
#else
  #include <vector>
  #include <memory>
  #include <numeric>
#endif

typedef int int32;

struct Point {
    int32 x, y;
};

struct BBox {
    Point ll, ur;
};

class NearestPointImageComputer {

  public:
    void compute(BBox bbox, int32 n_points, const Point* points, int32* image,
                 double* dist_retval = nullptr);
    void compute1(int32 n_points, const int32* points, const int32* p_tags,
                  int32 width, int32 stride, int32* f_values, int32* f_tags);
    void compute2(int32 n, int32* f_values, int32* f_tags);
    void compute2revised(int32 n, int32* f_values, int32* f_tags, int32 max_val);

  private:
    // centers of parabolas defining lower envelope
    std::vector<int32> m_centers;
    // left boundaries between parabolas, parabola [i] is minimal in [lbounds[i], lbounds[i+1])
    std::vector<int32> m_lbounds;
};

#endif
