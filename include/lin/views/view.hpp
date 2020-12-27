// vim: set tabstop=2:softtabstop=2:shiftwidth=2:expandtab

/** @file lin/views/view.hpp
 *  @author Kyle Krol
 */

#ifndef LIN_VIEWS_VIEW_HPP_
#define LIN_VIEWS_VIEW_HPP_

#include "const_matrix_view.hpp"
#include "const_vector_view.hpp"
#include "matrix_view.hpp"
#include "vector_view.hpp"

#include "../core.hpp"

#include <type_traits>

namespace lin {

/** @brief Creates a tensor view with traits based on the provided type.
 *
 *  @tparam C Tensor types the view's traits are taken from.
 *
 *  @param elems Constant element backing array.
 *  @param r     Initial row count.
 *  @param c     Initial column count.
 *
 *  @returns Constant matrix view.
 *
 *  This overload is chosen if the backing array contains constant elements and
 *  the traits provided describe a matrix.
 *
 *  Lin assertions errors will be triggered if the requested dimensions aren't
 *  possible given the tensor's traits.
 *
 *  @sa internal::traits
 *  @sa internal::is_matrix
 *
 *  @ingroup VIEWS
 */
template <class C, std::enable_if_t<internal::conjunction<
    internal::has_traits<C>, internal::is_matrix<C>>::value, size_t> = 0>
constexpr auto view(typename C::Traits::elem_t const *elems, size_t r = C::Traits::max_rows, size_t c = C::Traits::max_cols) {
  return internal::ConstMatrixView<typename C::Traits::elem_t, C::Traits::rows, C::Traits::cols, C::Traits::max_rows, C::Traits::max_cols>(elems, r, c);
}

/** @brief Creates a tensor view with traits based on the provided type.
 *
 *  @tparam C Tensor types the view's traits are taken from.
 *
 *  @param elems Constant element backing array.
 *  @param n     Initial length.
 *
 *  @returns Constant vector view.
 *
 *  This overload is chosen if the backing array contains constant elements and
 *  the traits provided describe a column vector.
 *
 *  Lin assertions errors will be triggered if the requested dimensions aren't
 *  possible given the tensor's traits.
 *
 *  @sa internal::traits
 *  @sa internal::is_col_vector
 *
 *  @ingroup VIEWS
 */
template <class C, std::enable_if_t<internal::conjunction<
    internal::has_traits<C>, internal::is_col_vector<C>>::value, size_t> = 0>
constexpr auto view(typename C::Traits::elem_t const *elems, size_t n = C::VectorTraits::max_length) {
  return internal::ConstVectorView<typename C::VectorTraits::elem_t, C::VectorTraits::length, C::VectorTraits::max_length>(elems, n);
}

/** @brief Creates a tensor view with traits based on the provided type.
 *
 *  @tparam C Tensor types the view's traits are taken from.
 *
 *  @param elems Constant element backing array.
 *  @param n     Initial length.
 *
 *  @returns Constant row vector view.
 *
 *  This overload is chosen if the backing array contains constant elements and
 *  the traits provided describe a row vector.
 *
 *  Lin assertions errors will be triggered if the requested dimensions aren't
 *  possible given the tensor's traits.
 *
 *  @sa internal::traits
 *  @sa internal::is_row_vector
 *
 *  @ingroup VIEWS
 */
template <class C, std::enable_if_t<internal::conjunction<
    internal::has_traits<C>, internal::is_row_vector<C>>::value, size_t> = 0>
constexpr auto view(typename C::Traits::elem_t const *elems, size_t n = C::VectorTraits::max_length) {
  return internal::ConstRowVectorView<typename C::VectorTraits::elem_t, C::VectorTraits::length, C::VectorTraits::max_length>(elems, n);
}

/** @brief Creates a tensor view with traits based on the provided type.
 *
 *  @tparam C Tensor types the view's traits are taken from.
 *
 *  @param elems Element backing array.
 *  @param r     Initial row count.
 *  @param c     Initial column count.
 *
 *  @returns Matrix view.
 *
 *  This overload is chosen if the backing array contains writable elements and
 *  the traits provided describe a matrix.
 *
 *  Lin assertions errors will be triggered if the requested dimensions aren't
 *  possible given the tensor's traits.
 *
 *  @sa internal::traits
 *  @sa internal::is_matrix
 *
 *  @ingroup VIEWS
 */
template <class C, std::enable_if_t<internal::conjunction<
    internal::has_traits<C>, internal::is_matrix<C>>::value, size_t> = 0>
constexpr auto view(typename C::Traits::elem_t *elems, size_t r = C::Traits::max_rows, size_t c = C::Traits::max_cols) {
  return internal::MatrixView<typename C::Traits::elem_t, C::Traits::rows, C::Traits::cols, C::Traits::max_rows, C::Traits::max_cols>(elems, r, c);
}

/** @brief Creates a tensor view with traits based on the provided type.
 *
 *  @tparam C Tensor types the view's traits are taken from.
 *
 *  @param elems Element backing array.
 *  @param n     Initial length.
 *
 *  @returns Vector view.
 *
 *  This overload is chosen if the backing array contains writable elements and
 *  the traits provided describe a column vector.
 *
 *  Lin assertions errors will be triggered if the requested dimensions aren't
 *  possible given the tensor's traits.
 *
 *  @sa internal::traits
 *  @sa internal::is_col_vector
 *
 *  @ingroup VIEWS
 */
template <class C, std::enable_if_t<internal::conjunction<
    internal::has_traits<C>, internal::is_col_vector<C>>::value, size_t> = 0>
constexpr auto view(typename C::Traits::elem_t *elems, size_t n = C::VectorTraits::max_length) {
  return internal::VectorView<typename C::VectorTraits::elem_t, C::VectorTraits::length, C::VectorTraits::max_length>(elems, n);
}

/** @brief Creates a tensor view with traits based on the provided type.
 *
 *  @tparam C Tensor types the view's traits are taken from.
 *
 *  @param elems Element backing array.
 *  @param n     Initial length.
 *
 *  @returns Row vector view.
 *
 *  This overload is chosen if the backing array contains writable elements and
 *  the traits provided describe a row vector.
 *
 *  Lin assertions errors will be triggered if the requested dimensions aren't
 *  possible given the tensor's traits.
 *
 *  @sa internal::traits
 *  @sa internal::is_row_vector
 *
 *  @ingroup VIEWS
 */
template <class C, std::enable_if_t<internal::conjunction<
    internal::has_traits<C>, internal::is_row_vector<C>>::value, size_t> = 0>
constexpr auto view(typename C::Traits::elem_t *elems, size_t n = C::VectorTraits::max_length) {
  return internal::RowVectorView<typename C::VectorTraits::elem_t, C::VectorTraits::length, C::VectorTraits::max_length>(elems, n);
}
}  // namespace lin

#endif
