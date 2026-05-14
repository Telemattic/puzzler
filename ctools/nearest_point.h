#ifndef NEAREST_POINT_H
#define NEAREST_POINT_H

#ifdef _DEBUG
  #undef _DEBUG
  #include <vector>
  #include <algorithm>
  #define _DEBUG
#else
  #include <vector>
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
    void compute(BBox bbox, int32 n_points, const Point* points, int32* image);
    void compute1(int32 n_points, int32* points, int32* p_tags,
                  int32 width, int32 stride, int32* f_values, int32* f_tags);
    void compute2(int32 n, int32* f_values, int32* f_tags);

  private:
    // centers of parabolas defining lower envelope
    std::vector<int32> m_centers;
    // left boundaries between parabolas, parabola [i] is minimal in [lbounds[i], lbounds[i+1])
    std::vector<int32> m_lbounds;
};

#endif
