// Cytosim was created by Francois Nedelec. Copyright 2007-2017 EMBL.
#include <cmath>
#include "capping_fiber_prop.h"
#include "capping_fiber.h"
#include "property_list.h"
#include "simul_prop.h"
#include "exceptions.h"
#include "glossary.h"
#include "simul.h"


Fiber* CappingFiberProp::newFiber() const
{
    return new CappingFiber(this);
}


void CappingFiberProp::clear()
{
    FiberProp::clear();
    
    for ( int i = 0; i < 2; ++i )
    {
        growing_force[i]   = INFINITY;
        growing_speed[i]   = 0;
        shrinking_speed[i] = 0;
        capping_rate[i]    = 0;
        uncapping_rate[i]  = 0;
        capped[i]          = false;
    }
}


void CappingFiberProp::read(Glossary& glos)
{
    FiberProp::read(glos);
    
    glos.set(growing_speed,   2, "growing_speed");
    glos.set(growing_force,   2, "growing_force");
    glos.set(shrinking_speed, 2, "shrinking_speed");
    glos.set(capping_rate,    2, "capping_rate");
    glos.set(uncapping_rate,  2, "uncapping_rate");
    glos.set(capped,          2, "capped");
}


void CappingFiberProp::complete(Simul const& sim)
{
    FiberProp::complete(sim);

    for ( int i = 0; i < 2; ++i )
    {
        if ( growing_force[i] <= 0 )
            throw InvalidParameter("fiber:growing_force should be > 0");
        if ( growing_speed[i] < 0 )
            throw InvalidParameter("fiber:growing_speed should be >= 0");
        
        growing_speed_dt[i]   = growing_speed[i] * sim.time_step();
        
        if ( shrinking_speed[i] > 0 )
            throw InvalidParameter("fiber:shrinking_speed should be <= 0");
        shrinking_speed_dt[i] = shrinking_speed[i] * sim.time_step();

        if ( capping_rate[i] < 0 )
            throw InvalidParameter("fiber:capping_rate should be >= 0");
        capping_rate_dt[i] = capping_rate[i] * sim.time_step();

        if ( uncapping_rate[i] < 0 )
            throw InvalidParameter("fiber:capping_rate should be >= 0");
        uncapping_rate_dt[i] = uncapping_rate[i] * sim.time_step();
    }
}


void CappingFiberProp::write_values(std::ostream& os) const
{
    FiberProp::write_values(os);
    write_value(os, "growing_force",   growing_force,   2);
    write_value(os, "growing_speed",   growing_speed,   2);
    write_value(os, "shrinking_speed", shrinking_speed, 2);
    write_value(os, "capping_rate",    capping_rate,    2);
    write_value(os, "uncapping_rate",  capping_rate,    2);
    write_value(os, "capped",          capped,          2);
}
