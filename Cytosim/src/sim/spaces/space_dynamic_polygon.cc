// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.


#include "dim.h"
#include "space_dynamic_polygon.h"
#include "space_dynamic_prop.h"
#include "exceptions.h"
#include "mecapoint.h"
#include "glossary.h"
#include "polygon.h"
#include "random.h"
#include "meca.h"
#include <fstream>
#include <numeric>
#include <cmath>
#include <vector>


SpaceDynPolygon::SpaceDynPolygon(SpaceDynamicProp const* p)
: Space(p)
{
    inf_.reset();
    sup_.reset();
    surface_ = 0;
    height_ = 0;
    reset_forces();
   
    if ( DIM == 1 )
        throw InvalidParameter("polygon is not usable in 1D");
}


SpaceDynPolygon::~SpaceDynPolygon()
{
}

//------------------------------------------------------------------------------
#pragma mark - I/O

/**
 recalculate bounding box, volume
 and points offsets that are used to project
 */
void SpaceDynPolygon::resize(Glossary& opt)
{
    unsigned ord = 40;
    std::string file;
    
    if ( opt.set(file, "file") )
        poly_.read(file);
    else if ( !prop()->dimensions.empty() )
        poly_.read(prop()->dimensions);
    else if ( opt.has_key("points") )
    {
        // specify vertices directly:
        unsigned nbp = (unsigned)opt.num_values("points");
        poly_.allocate(nbp);
        for ( unsigned p = 0; p < nbp; ++p )
        {
            Vector2 vec(0,0);
            if ( ! opt.set(vec, "points", p) )
                throw InvalidParameter("polygon:points must be a list of comma-separated points: X Y, X Y, X Y, etc.");
            poly_.setPoint(p, vec.XX, vec.YY);
        }

        resize_force(nbp); // resize forces to fit number of polygon vertices
    }
    else if ( opt.set(ord, "order") )
    {
        real rad, ang = 0;
        opt.set(rad, "radius");
        opt.set(ang, "angle");
        poly_.set(ord, rad, ang);

        resize_force(ord);

    }
    else
        return;
    
    real x;
    if ( opt.set(x, "scale") )
        poly_.scale(x, x);

    if ( opt.set(x, "inflate") )
        poly_.inflate(x);
    
#if ( DIM == 3 )
    x = height_;
    if ( opt.set(x, "height") )
        x *= 0.5;
    if ( x < 0 )
        throw InvalidParameter("polygon:height must be >= 0");
    height_ = x;
#endif

    update();

}

void SpaceDynPolygon::resize_force(unsigned num_vertices) // Initialize with zero forces
{
    vertex_forces.resize(num_vertices);
    reset_forces();
}

//-----------------------------------Energy and Forces section of space dynamic polygon-------------------------------------------//
// Calculate the energy and forces from osmotic pressure acting on each vertex the polygon using van't hoff approximation 
// for osmotic pressure and assuming a preferred volume V_bar for the polygon


std::vector<Vector2> SpaceDynPolygon::calculateOsmoticForces() const {

    real Kv = prop()->Kv; //osmotic strength in units of nN.um;

    real V_bar = prop()->V_bar; //preferred volume in units of um^3;

    real volume = abs_real(poly_.surface());

    if(V_bar <= 0)
    {
        
        V_bar = volume; // if preferred volume is not set, use current volume
    }

    std::vector<Vector2> osmoticForce(poly_.nbPoints());
    osmoticForce.assign(poly_.nbPoints(), Vector2(0,0));

    size_t n = poly_.nbPoints();
    
    Vector2 d;
    std::vector<real> edgeLengths(n);
    std::vector<Vector2> edge_normal(n);

    for (size_t i = 0; i < n; ++i) {


        d.XX = poly_.pts_[(i + 1) % n].xx - poly_.pts_[i].xx;
        d.YY = poly_.pts_[(i + 1) % n].yy - poly_.pts_[i].yy;
        edgeLengths[i] = d.norm();

        //Calculate the edge normals. Assumes a clockwise orientation convention of the edge normal vectors
        edge_normal[i].XX =  d.YY / edgeLengths[i];
        edge_normal[i].YY = -1 * d.XX / edgeLengths[i];  
    }

    for (size_t i = 0; i < n; ++i) 
    {
        int j = i - 1;
        if (i == 0)
        {
            j = n - 1;
        }

        Vector2 vertex_normal = 0.5 * ((edge_normal[i] * edgeLengths[i]) + (edge_normal[j] * edgeLengths[j]));

        osmoticForce[i] = Kv * ((1.0/volume) - (1.0/V_bar)) * vertex_normal;

    }

  return osmoticForce;
}


// Calculate the forces from mechanical surface tension acting on each vertex of the polygon
std::vector<Vector2> SpaceDynPolygon::calculateTensionForces() const {
  double Ksg = prop()->tension;
  const auto &vertex_positions = poly_.pts_;

  std::vector<Vector2> tensionForce(poly_.nbPoints());
  tensionForce.assign(poly_.nbPoints(), Vector2(0,0));

  Vector2 d, edgeUnitVector;
  real edgeLength;
  for (size_t i = 0; i < poly_.nbPoints(); ++i) {
    d.XX = vertex_positions[(i + 1)].xx - vertex_positions[i].xx;
    d.YY = vertex_positions[(i + 1)].yy - vertex_positions[i].yy;
    edgeLength = d.norm();
    edgeUnitVector = d / edgeLength;

    tensionForce[i] += -2 * Ksg * edgeLength * (-1 * edgeUnitVector);
    if (i == poly_.nbPoints() - 1) {
      tensionForce[0] += -2 * Ksg * edgeLength * (edgeUnitVector);
    } else {
      tensionForce[i + 1] += -2 * Ksg * edgeLength * (edgeUnitVector);
    }
  }

  return tensionForce;
}

std::vector<Vector2> SpaceDynPolygon::calculateRegularizationForces() const {
  double Ksl = 20.0; // regularization strength in units of mN.um;
  const auto &vertex_positions = poly_.pts_;

  size_t n = poly_.nbPoints();

  std::vector<Vector2> regForce(n);
  regForce.assign(n, Vector2(0,0));

  Vector2 d;
  std::vector<Vector2>edgeUnitVector(n);
  std::vector<real> edgeLength(n);
  real referenceLength = 0.0;

  for (size_t i = 0; i < n; ++i) {
    d.XX = vertex_positions[(i + 1)].xx - vertex_positions[i].xx;
    d.YY = vertex_positions[(i + 1)].yy - vertex_positions[i].yy;
    edgeLength[i] = d.norm();
    edgeUnitVector[i] = d / edgeLength[i];
    referenceLength += edgeLength[i];
  }

  int cnt = edgeLength.size();
  referenceLength = referenceLength / cnt; // average edge length 

  for (size_t i = 0; i < n; ++i) {

    regForce[i] += -2 * Ksl * (-1 * edgeUnitVector[i]) * (edgeLength[i] - referenceLength) / (referenceLength * referenceLength);
    if (i == n - 1) {
        regForce[0] += -2 * Ksl * edgeUnitVector[i] * (edgeLength[i] - referenceLength) / (referenceLength * referenceLength);
    } else {
        regForce[i + 1] += -2 * Ksl * edgeUnitVector[i] * (edgeLength[i] - referenceLength) / (referenceLength * referenceLength);
    }
 }

  return regForce;
}


std::vector<Vector2> SpaceDynPolygon::calculateBendingForces_optimized() const
{
    double Kb = prop()->bending / 4;

    size_t n = poly_.nbPoints(); // vertex_positions.size();
   
    std::vector<Vector2> bendingForce(n); 
    bendingForce.assign(bendingForce.size(), Vector2(0,0));

    Vector2 d_pos;
    std::vector<real> edgeLengths(n);
    std::vector<Vector2> edge_normal(n);
    std::vector<real> edgeAbsoluteAngles(n);
    std::vector<Vector2> edgeUnitVectors(n); 
    std::vector<real> tan_vertex_turning_angles(n);
    std::vector<real> cotan_vertex_turning_angles(n);

    for (size_t i = 0; i < n; ++i) {
        d_pos.XX = poly_.pts_[(i + 1) % n].xx - poly_.pts_[i].xx;
        d_pos.YY = poly_.pts_[(i + 1) % n].yy - poly_.pts_[i].yy;

        edgeLengths[i] = d_pos.norm();

        //Calculate the edge normals. Assumes a clockwise orientation convention of the edge normal vectors
        edge_normal[i].XX = d_pos.YY / (edgeLengths[i] * edgeLengths[i]);
        edge_normal[i].YY = -1 * d_pos.XX / (edgeLengths[i] * edgeLengths[i]); 
       
        

        edgeAbsoluteAngles[i] = std::atan2(d_pos.YY, d_pos.XX);
        edgeUnitVectors[i] = (1 / edgeLengths[i]) * d_pos;
        
    }

    for (size_t i = 0, j = n-1; i < n; ++i, ++j) {
        if (j == n) 
        {
            j = 0;
        }
        real vertexTurningAngle = fmod(edgeAbsoluteAngles[j] - edgeAbsoluteAngles[i], 2 * M_PI);
        vertexTurningAngle = fmod(vertexTurningAngle + M_PI, 2 * M_PI) - M_PI;

        tan_vertex_turning_angles[i] = std::tan(vertexTurningAngle / 2);
        cotan_vertex_turning_angles[i] = std::pow(cos(vertexTurningAngle / 2), -2);  

    }

    for (size_t i = 0, j = 1; i < n; ++i, ++j) {
        if (j == n) 
            j = 0;

        real edgeCurvature = (tan_vertex_turning_angles[i] + tan_vertex_turning_angles[j]) / edgeLengths[i]; 


        Vector2 dki_squared_j = 2 * edgeLengths[i] * edgeCurvature * (0.5 * ((cotan_vertex_turning_angles[i] * -1 * edge_normal[i])) - (edgeCurvature * -1 * edgeUnitVectors[i])); // + (cotan_vertex_turning_angles2[i] * edge_normal2[i])
        Vector2 dli_j = edgeCurvature * edgeCurvature * -1 * 2 * edgeLengths[i] * edgeUnitVectors[i];
        
        Vector2 dki_squared_j1 = 2 * edgeLengths[i] * edgeCurvature * ((0.5 * ((cotan_vertex_turning_angles[i] * (edge_normal[i] + edge_normal[j])) + (cotan_vertex_turning_angles[j] * edge_normal[j]))) - (edgeCurvature * edgeUnitVectors[i]));
        Vector2 dli_j1 = edgeCurvature * edgeCurvature * 2 * edgeLengths[i] * edgeUnitVectors[i];

        bendingForce[i] += -1 * Kb * (dki_squared_j + dli_j);

        if(i == n - 1)
        {
            bendingForce[0] += -1 * Kb * (dki_squared_j1 + dli_j1);
        }
        else
        {
            bendingForce[i+1] += -1 * Kb * (dki_squared_j1 + dli_j1);
        }

    }   
    return bendingForce;
}




void SpaceDynPolygon::update()
{
    surface_ = poly_.surface();
    if ( surface_ < 0 )
    {
        poly_.flip();
        surface_ = poly_.surface();
    }
    assert_true( surface_ > 0 );
    
    if ( poly_.complete(REAL_EPSILON) )
        throw InvalidParameter("unfit polygon: consecutive points may overlap");

    real box[4];
    poly_.find_extremes(box);
    inf_.set(box[0], box[2], -height_);
    sup_.set(box[1], box[3],  height_);

}

void SpaceDynPolygon::step() {
  std::vector<Vector2> step(
      prop()->mobility_dt *
      (calculateTensionForces() + calculateBendingForces_optimized() 
       + calculateRegularizationForces() + calculateOsmoticForces() + vertex_forces));

  for (unsigned i = 0; i < poly_.nbPoints(); ++i) {
    poly_.pts_[i].xx += step[i].XX;
    poly_.pts_[i].yy += step[i].YY;
  }

  poly_.wrap();
  reset_forces();
}

bool SpaceDynPolygon::inside(Vector const& W) const
{
#if ( DIM > 2 )
    if ( abs_real(W.ZZ) > height_ )
        return false;
#endif
#if ( DIM > 1 )
    return poly_.inside(W.XX, W.YY, 1);
#else
    return false;
#endif
}


Vector SpaceDynPolygon::place() const
{
    if ( surface_ <= 0 )
        throw InvalidParameter("cannot pick point inside polygon of null surface");
    return Space::place();
}


Vector SpaceDynPolygon::project(Vector const& W) const
{
    Vector P(W);
#if ( DIM == 2 )
    
    unsigned hit;
    poly_.project(W.XX, W.YY, P.XX, P.YY, hit);
    
#elif ( DIM > 2 )
    
    if ( abs_real(W.ZZ) > height_ )
    {
        if ( poly_.insideWinding(W.XX, W.YY) )
        {
            // too high or too low in the Z axis, but inside XY
            P.XX = W.XX;
            P.YY = W.YY;
        }
        else
        {
            // outside in Z and XY
            unsigned hit;
            poly_.project(W.XX, W.YY, P.XX, P.YY, hit);
        }
        P.ZZ = std::copysign(height_, W.ZZ);
    }
    else
    {
        unsigned hit;
        poly_.project(W.XX, W.YY, P.XX, P.YY, hit);
        if ( poly_.insideWinding(W.XX, W.YY) )
        {
            // inside in the Z axis and the XY polygon:
            // to the polygonal edge in XY plane:
            real HH = (W.XX-P.XX)*(W.XX-P.XX) + (W.YY-P.YY)*(W.YY-P.YY);
            // to the top/bottom plates:
            real V = height_ - abs_real(W.ZZ);
            // compare distances
            if ( V * V < HH )
                return Vector(W.XX, W.YY, std::copysign(height_, W.ZZ));
        }
        P.ZZ = W.ZZ;
    }
    
#endif
    return P;
}

//------------------------------------------------------------------------------
#pragma mark - setConfinement

void SpaceDynPolygon::setInteractions(Meca&, Simul const&)
{
    reset_forces();
}

void SpaceDynPolygon::setConfinement(Vector const&pos, Mecapoint const& mp,
                                         Meca& meca, real stiff) const
{
    real cutoff = 0.0035; // cutoff for steric interactions between membrane and fibers, based on typical fiber radius of 3.5 nm
    Vector prj;
    prj = project(pos);
    Vector dir = pos - prj;
    real n = dir.normSqr();
    
    size_t npoints = poly_.nbPoints();
    Vector2 P(pos.XX, pos.YY);
    int inside = poly_.insideWinding(P.XX, P.YY);

    for (unsigned j = 0; j < npoints; ++j)
        {   

            real x = P.XX - poly_.pts_[j].xx;
            real y = P.YY - poly_.pts_[j].yy;
            
            real a = poly_.pts_[j].dx * x + poly_.pts_[j].dy * y;

            if (a < 0)
            {
                // projection is before the segment
                a = 0;
            }
            if (a > poly_.pts_[j].len)
            {
                // projection is after the segment
                a = poly_.pts_[j].len;
            }

            Vector proj(poly_.pts_[j].xx + a * poly_.pts_[j].dx, poly_.pts_[j].yy + a * poly_.pts_[j].dy, 0.0);
            
            // distance to the segment:
            Vector b = pos - proj;
            real da = b.norm();

            Vector2 b2 = Vector2(b.XX, b.YY);

            real len = poly_.pts_[j].len;
            real len1 = (len - a)/len;
            real len2 = a/len;

            if ( da > 0 && da < cutoff )
            {
                if (inside == 0)
                {
                    // Register the force to the membrane edge
                    decomposeForce(len1 * stiff * b2, j);

                    if(j == npoints - 1)
                    {
                        decomposeForce(len2 * stiff * b2, 0);
                    }
                    else
                    {
                        decomposeForce(len2 * stiff * b2, j+1);
                    }
                }
            }
            //And to the meca
            meca.addPlaneClamp(mp, prj, dir, stiff/n);
        }
    
}


//------------------------------------------------------------------------------
#pragma mark - I/O

void SpaceDynPolygon::write(Outputter &out) const {
    
    writeMarker(out, TAG);
    writeShape(out, "dynamic_polygon");
    
    out.writeUInt32(poly_.nbPoints());
    for (int i = 0; i < poly_.nbPoints(); i++) {
        out.writeFloat(poly_.pts_[i].xx);
        out.writeFloat(poly_.pts_[i].yy);
        // currently don't print out info
        // out.writeInt32(poly_.pts_[i].info);
    }
}

void SpaceDynPolygon::read(Inputter &in, Simul &, ObjectTag) {
    
    std::string str, str2;
    str = in.get_characters(16); // stored as 16 characters

    // check that this matches current Space:
    if (str.compare(0, 15, "dynamic_polygon")) {
        std::ostringstream oss;
        oss << "Space `" << prop()->name() << "' has shape " << str;
        oss << " in objects and " << prop()->shape << " in property";
        throw InvalidIO(oss.str());
    }


    poly_.npts_ = in.readUInt32();
    poly_.allocate(poly_.npts_);
    unsigned int n = poly_.npts_;

    // Read in 2*n corresponding to x1, y1, x2, y2 .. xn, yn
    for (int i = 0; i < poly_.npts_; ++i) {
        poly_.pts_[i].xx = in.readFloat();
        poly_.pts_[i].yy = in.readFloat();
    }

    update();
    resize_force(n); // resize forces to fit number of polygon vertices

}

void SpaceDynPolygon::setLengths(const real len[8])
{
    height_ = len[0];
}

void SpaceDynPolygon::report(std::ostream& os) const 
{   
    for (int i = 0; i < poly_.nbPoints(); i++) {
        os << "\n" << poly_.pts_[i].xx <<"\t" << poly_.pts_[i].yy;

    }
}

//------------------------------------------------------------------------------
#pragma mark - OpenGL display

#ifdef DISPLAY

#include "gle.h"
#include "gym_flute.h"
#include "gym_draw.h"

void SpaceDynPolygon::drawPolygon(float lines, float points) const
{
    const unsigned nbp = poly_.nbPoints();
    Polygon::Point2D const* pts = poly_.pts_;
    flute2 * flt = gym::mapBufferV2(nbp+1);
    for ( size_t n = 0; n <= nbp; ++n )
    {
        std::clog << "flute2["<<n<<"] = "<<pts[n].xx<<", "<<pts[n].yy<<std::endl;
        flt[n].set(pts[n].xx, pts[n].yy);
    }
    gym::unmapBufferV2();
    
    glEnable(GL_STENCIL_TEST);
    glClearStencil(1);
    glClear(GL_STENCIL_BUFFER_BIT);
    glStencilFunc(GL_EQUAL, 1, ~0U);
    glStencilOp(GL_KEEP, GL_ZERO, GL_ZERO);
    if ( lines > 0 )
    {
        gym::drawLineStrip(lines, 0, nbp+1);
    }
    if ( points > 0 )
    {
        gym::drawPoints(points, 0, nbp);
    }
    glClear(GL_STENCIL_BUFFER_BIT);
    glDisable(GL_STENCIL_TEST);
}

void SpaceDynPolygon::draw3D() const
{
    const float H(-height_);
    const unsigned nbp = poly_.nbPoints();
    Polygon::Point2D const* pts = poly_.pts_;
    flute3 * flt = gym::mapBufferV3(2*nbp+2);
    for ( size_t i = 0; i <= nbp; ++i )
    {
        float X(pts[i].xx), Y(pts[i].yy);
        flt[2*i  ] = { X, Y, -H };
        flt[2*i+1] = { X, Y,  H };
    }
    gym::unmapBufferV3();
    // display sides
    gym::drawTriangleStrip(0, 2*nbp+2);

    float lines = 2;
    if ( lines )
    {
        // display bottom
        gym::rebindBufferV3(2, 0);
        gym::drawLineStrip(lines, 0, nbp+1);
        // display top
        gym::rebindBufferV3(2, 1);
        gym::drawLineStrip(lines, 0, nbp+1);
    }
    gym::cleanupV();
}

#else

void SpaceDynPolygon::drawPolygon(float, float) const {}
void SpaceDynPolygon::draw3D() const {}

#endif
