// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.
#ifndef SPACE_DYNAMIC_POLYGON_H
#define SPACE_DYNAMIC_POLYGON_H

#include "dim.h"
#include "space.h"
#include "polygon.h"
#include "space_dynamic_prop.h"
#include "meca.h"
#include "mecable.h"

inline std::vector<Vector2> operator+(const std::vector<Vector2> &v1,
                                      const std::vector<Vector2> &v2) {
  assert_true(v1.size() == v2.size());
  std::vector<Vector2> sum(v1.size());
  for (size_t i = 0; i < v1.size(); ++i) {
    sum[i] = v1[i] + v2[i];
  }
  return sum;
}

inline std::vector<Vector2> operator*(real x, const std::vector<Vector2> &v) {
  std::vector<Vector2> res(v.size());
  for (size_t i = 0; i < v.size(); ++i) {
    res[i] = x * v[i];
  }
  return res;
}

/// A deformable 2-D polygon around the Z axis
/**
 
 The coordinates of the 2D polygon (X Z) can be determined from a radius and number of points or can be read from a file.

 @ingroup SpaceGroup
*/

class SpaceDynPolygon : public Space
{
private:
    
    /// The 2D polygon
    Polygon poly_;

    mutable std::vector<Vector2> vertex_forces; 
    
    /// pre-calculated bounding box since this is called often
    Vector inf_, sup_;
    
    /// Volume calculated from polygon
    real surface_;
    
    /// half the total height in Z
    real height_;


    void reset_forces(){

        vertex_forces.assign(vertex_forces.size(), Vector2(0,0));
    }

    //Add forces to vertices: takes in a vector2 force and a vertex index
    void decomposeForce(const Vector2& force, unsigned pos) const {
        vertex_forces[pos] += force;
    }

    /// update polygon shape and bounding box after vertices have been moved
    void update();


public:

    ///creator
    SpaceDynPolygon(SpaceDynamicProp const*);
    
    ///destructor
    ~SpaceDynPolygon();
    
    /// Property
    SpaceDynamicProp const* prop() const { return static_cast<SpaceDynamicProp const*>(Space::prop); }

    /// change dimensions
    void resize(Glossary& opt);

    //resize force vectors
    void resize_force(unsigned num_vertices);

    /// return bounding box in `inf` and `sup`
    void boundaries(Vector& inf, Vector& sup) const { inf=inf_; sup=sup_; }
    
    
    /// the volume inside - 2D version
    real volume2D() const;

    /// true if the point is inside the Space
    bool inside(Vector const&) const;
    
    /// return point on the edge that is closest to `pos`
    Vector project(Vector const& pos) const;

    /// a random position inside the volume
    Vector place() const;

    /// apply a force directed towards the edge of the Space
    void setConfinement(Vector const&pos, Mecapoint const& mp, Meca& meca, real stiff) const;
    
    /// add interactions between fibers and reentrant corners
    void setInteractions(Meca&, Simul const&);
    
    /// write to file
    void write(Outputter&) const;

    /// report to stream
    void report(std::ostream&) const;

    /// get dimensions from array `len`
    void setLengths(const real len[8]);

    /// read from file
    void read(Inputter&, Simul&, ObjectTag);

    /// estimate Volume using a crude Monte-Carlo method with `cnt` calls to Space::inside()
    real estimateVolumeZ(size_t cnt) const;

    //Update the polygon shape
    void step();

    /// OpenGL display function
    void drawPolygon(float, float) const;

    /// OpenGL display function
    void draw3D() const;

    /// OpenGL display function
    void draw2D(float) const { draw3D(); }

    /// Test membrane mechanics function
    std::tuple<double, double> _energy() const; // _energy(const std::vector<std::array<double, 2>>& vertex_positions)

    //calculate forces using ddg derivative with respect to vertex positions

    std::vector<Vector2> calculateOsmoticForces() const;

    std::vector<Vector2> calculateTensionForces() const;

    std::vector<Vector2> calculateRegularizationForces() const;

    std::vector<Vector2> calculateBendingForces() const;

    std::vector<Vector2> calculateBendingForces_optimized() const;
};

#endif