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

/// an axisymmetric volume obtained by rotating a polygon around the Z axis
/**
 This is only valid in 3D.
 The volume is built by rotating a closed 2D polygon around the Z axis.
 
 The coordinates of the 2D polygon (X Z) are read from a file.
 The offset `shift` is added to the X-coordinate before the polygon is rotated around Z.
 Volume is estimated by Monte-Carlo, and takes an instant.
 
 Parameters:
     - file: name of file with polygon data
    .

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

    //Pointer to a mecable object (in this case fiber segment)
    // Mecable* mec_;

    //Index of the point of interest
    // unsigned pti_;

    //mobility
    // real mobility_dt;

    void reset_forces() const {

        vertex_forces.assign(vertex_forces.size(), Vector2(0,0));

        // for( auto v : vertex_forces){
        //     v = Vector2(0,0);
        // }
        // Use in C++20
        // std::fill(vertex_forces.begin(), vertex_forces.end(), {0,0});
    }

    //Add forces to vertices: takes in a vector2 force and a vertex index
    void decomposeForce(const Vector2& force, unsigned pos) const {
        vertex_forces[pos] += force;
    }

    /**
     Closest point of polygon edge `j` to `P`.
     */
    real edgeProjection(unsigned j, Vector2 const& P, Vector2& b, real& da) const;

    /// mean edge length, the length scale the adaptive sub-step is measured in
    real meanEdge() const;


    /// update data structure
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
    
    /// the volume inside
    // real volume() const { return volume_; }
    
    /// the volume inside - 2D version
    real volume2D() const;

    /// true if the point is inside the Space
    bool inside(Vector const&) const;
    
    /// return point on the edge that is closest to `pos`
    Vector project(Vector const& pos) const;

    /**
     Signed distance to the membrane: >= 0 inside, negative outside.
     */
    real depthBelowMobileEdge(Vector const& pos) const;

    /**
     Exact OUTWARD normal, overriding Space::normalToEdge().
     */
    Vector normalToEdge(Vector const& pos) const;


    /// a random position inside the volume
    Vector place() const;

    /// apply a force directed towards the edge of the Space
    void setConfinement(Vector const&pos, Mecapoint const& mp, Meca& meca, real stiff) const;


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

    ///compute osmotic energy
    // real _osmotic_energy() const;

    /// OpenGL display function
    void drawPolygon(float, float) const;

    /// OpenGL display function
    void draw3D() const;

    /// OpenGL display function
    void draw2D(float) const { draw3D(); }

    /// Test membrane mechanics function
    std::tuple<double, double> _energy() const; // _energy(const std::vector<std::array<double, 2>>& vertex_positions)

    ///compute energy of the membrane using Helfrich Hamiltonian
    // real _energy(autodiff::ArrayXreal&) const;

    ///compute osmotic energy of the membrane using vant Hoff equation
    real _osmotic_energy() const;

    ///calculate forces using ddg derivative with respect to vertex positions

    std::vector<Vector2> calculateOsmoticForces() const;

    std::vector<Vector2> calculateTensionForces() const;

    std::vector<Vector2> calculateRegularizationForces() const;
    std::vector<Vector2> calculateBendingForces_optimized() const;

    /**
     Test to show that the energies match the force kernels above, term by term.
     */
    real energyTension() const;
    real energyBending() const;
    real energyRegularization() const;
    real energyOsmotic() const;

    /// sum of the four; the membrane's own potential energy
    real totalEnergy() const;

    /// finite-difference check that -dE/dx matches each force kernel
    void selfTest(std::ostream&) const;


};

#endif
