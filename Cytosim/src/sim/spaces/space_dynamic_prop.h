// Cytosim was created by Francois Nedelec. Copyright 2022 Cambridge University

#ifndef SPACE_DYNAMIC_PROP_H
#define SPACE_DYNAMIC_PROP_H

#include "space_prop.h"
#include "property.h"

class SpaceProp;
class Space;

class SpaceDynamicProp : public SpaceProp
{
        
public:
    
    /**
     @defgroup DynamicSpacePar Parameters of Dynamic Space
     @ingroup Parameters
     @{
    */

    /// Viscosity
    real viscosity;

    /// Viscosity for rotation
    real viscosity_rot;
    
    // tension of the polygon (units of pN um)
    real tension;

    // bending modulus of the polygon (units of pN um)
    real bending;

    // van't hoff index of the polygon (units of pN um)
    real Kv;

    // preferred volume of the polygon (units of um^3)
    real V_bar;
    
    real cutoff; // cutoff distance for steric interaction (units of um)
    
    // volume of the polygon (mutable because changed by const method)
    mutable real volume;
    
    /// @}

    /// derived values equal to timestep / viscosity
    real mobility_dt, mobility_rot_dt;

public:

    /// constructor
    SpaceDynamicProp(const std::string& n) : SpaceProp(n)  { clear(); }
    
    /// destructor
    ~SpaceDynamicProp() { }
	
    /// set from a Glossary
    virtual void read(Glossary&);
	
    /// check and derive more parameters
    virtual void complete(Simul const&);
	
    /// write all values
    virtual void write_values(std::ostream&) const;
    
    /// set default values
    virtual void clear();
    
    /// create a new, uninitialized, Space
    virtual Space * newSpace() const;
	
};

#endif

