// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.
#ifndef CAPPING_FIBER_PROP
#define CAPPING_FIBER_PROP

#include "fiber_prop.h"


/// additional Property for CappingFiber
/**
 @ingroup Properties
 Assembly is continuous, and can occur at both ends.
 */
class CappingFiberProp : public FiberProp
{
    friend class CappingFiber;
    
public:
    
    /**
     @defgroup CappingFiberPar Parameters of CappingFiber
     @ingroup Parameters
     Inherits @ref FiberPar.
     @{
     */
    
    /// see @ref CappingFiber

    /// Characteristic force for polymer assembly
    real    growing_force[2];
    
    /// Speed of assembly
    real    growing_speed[2];
    
    /// Speed of disassembly
    real    shrinking_speed[2];

    /// Rate of capping
    real    capping_rate[2];

    /// Rate of uncapping
    real    uncapping_rate[2];

    /// Capped state
    bool    capped[2];
    
    /// @}
    
private:
    
    /// derived variable:
    real    growing_speed_dt[2];
    /// derived variable:
    real    shrinking_speed_dt[2];
    /// derived variable:
    real    capping_rate_dt[2];
    /// derived variable:
    real    uncapping_rate_dt[2];

public:

    /// constructor
    CappingFiberProp(const std::string& n) : FiberProp(n) { clear(); }

    /// destructor
    ~CappingFiberProp() { }
    
    /// return a Fiber with this property
    Fiber* newFiber() const;
    
    /// set default values
    void clear();
       
    /// set using a Glossary
    void read(Glossary&);
   
    /// check and derive parameter values
    void complete(Simul const&);
    
    /// return a carbon copy of object
    Property* clone() const { return new CappingFiberProp(*this); }

    /// write
    void write_values(std::ostream&) const;

};

#endif
