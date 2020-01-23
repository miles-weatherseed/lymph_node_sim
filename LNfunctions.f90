subroutine inside_sphere(vec, radius, sts)

    implicit none
    double precision, dimension(3), intent(in) :: vec
    double precision, intent(in) :: radius
    logical, intent(out) :: sts

    if (NORM2(vec) < radius) then
        sts = .TRUE.
    else
        sts = .FALSE.
    end if

end subroutine inside_sphere

subroutine threed_to_1d(coord_vec, numPositions, oned)

    implicit none
    integer, dimension(3), intent(in) :: coord_vec
    integer, intent(in) :: numPositions
    integer, intent(out) :: oned

    oned = coord_vec(1)*numPositions**2 + coord_vec(2)*numPositions + coord_vec(3)

end subroutine threed_to_1d

subroutine set_coordinates(posn_vec, radius, cellSide, coord_vec)

    implicit none
    double precision, dimension(3), intent(in) :: posn_vec
    double precision, intent(in) :: radius
    double precision, intent(in) :: cellSide
    integer, dimension(3), intent(out) :: coord_vec

    coord_vec = INT((posn_vec + radius)/cellSide)

end subroutine set_coordinates